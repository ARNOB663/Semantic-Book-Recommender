import pandas as pd
import numpy as np
import gradio as gr

from langchain_community.document_loaders import TextLoader
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ----------------------------
# Load data
# ----------------------------
books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"].fillna("cover-not-found.jpg") + "&fife=w800"

# ----------------------------
# Load & split documents
# ----------------------------
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1,
    chunk_overlap=0
)

documents = text_splitter.split_documents(raw_documents)

# ----------------------------
# Embeddings
# ----------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db_books = Chroma.from_documents(
    documents=documents,
    embedding=embedding
)

# ----------------------------
# Recommendation logic
# ----------------------------
def retrieve_semantic_recommendations(
    query: str,
    category: str,
    tone: str,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)

    book_ids = []
    for rec in recs:
        try:
            book_ids.append(int(rec.page_content.strip('"').split()[0]))
        except:
            continue

    book_recs = books[books["isbn13"].isin(book_ids)].copy()

    # Category filter
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    # Tone sorting
    tone_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness"
    }

    if tone in tone_map:
        book_recs = book_recs.sort_values(by=tone_map[tone], ascending=False)

    return book_recs.head(final_top_k)

# ----------------------------
# Gradio callback
# ----------------------------
def recommend_books(query, category, tone):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        short_desc = " ".join(description.split()[:30]) + "..."

        authors = row["authors"].split(";")
        if len(authors) == 2:
            authors_str = f"{authors[0]} and {authors[1]}"
        elif len(authors) > 2:
            authors_str = f"{', '.join(authors[:-1])}, and {authors[-1]}"
        else:
            authors_str = authors[0]

        caption = f"{row['title']} by {authors_str}: {short_desc}"
        results.append((row["large_thumbnail"], caption))

    return results

# ----------------------------
# UI
# ----------------------------
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Describe a book you want to read",
            placeholder="e.g., A dark story about revenge and justice"
        )
        category_dropdown = gr.Dropdown(categories, value="All", label="Category")
        tone_dropdown = gr.Dropdown(tones, value="All", label="Emotional Tone")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## 🔍 Recommendations")
    output = gr.Gallery(columns=8, rows=2)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    dashboard.launch()
