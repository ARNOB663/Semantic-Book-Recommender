import pandas as pd
import numpy as np
import gradio as gr

from langchain_community.document_loaders import TextLoader
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

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    tone_map = {
        "😊 Happy": "joy",
        "😲 Surprising": "surprise",
        "😠 Angry": "anger",
        "😰 Suspenseful": "fear",
        "😢 Sad": "sadness"
    }

    if tone in tone_map:
        book_recs = book_recs.sort_values(by=tone_map[tone], ascending=False)

    return book_recs.head(final_top_k)

# ----------------------------
# Generate book cards HTML
# ----------------------------
def create_book_card(row):
    title = row["title"][:50] + "..." if len(row["title"]) > 50 else row["title"]
    
    authors = row["authors"].split(";")
    if len(authors) == 2:
        authors_str = f"{authors[0]} & {authors[1]}"
    elif len(authors) > 2:
        authors_str = f"{authors[0]} et al."
    else:
        authors_str = authors[0]
    authors_str = authors_str[:30] + "..." if len(authors_str) > 30 else authors_str
    
    description = " ".join(str(row["description"]).split()[:25]) + "..."
    rating = row.get("average_rating", "N/A")
    year = int(row.get("published_year", 0)) if pd.notna(row.get("published_year")) else "N/A"
    category = row.get("simple_categories", "Unknown")
    
    # Emotion badges
    emotions = {
        "joy": ("😊", "#4ade80"),
        "sadness": ("😢", "#60a5fa"),
        "anger": ("😠", "#f87171"),
        "fear": ("😰", "#a78bfa"),
        "surprise": ("😲", "#fbbf24")
    }
    
    top_emotion = max(emotions.keys(), key=lambda e: row.get(e, 0) if pd.notna(row.get(e, 0)) else 0)
    emotion_icon, emotion_color = emotions[top_emotion]
    emotion_score = row.get(top_emotion, 0)
    emotion_score = f"{emotion_score:.0%}" if pd.notna(emotion_score) else "N/A"
    
    return f'''
    <div class="book-card">
        <div class="book-cover-container">
            <img src="{row["large_thumbnail"]}" alt="{title}" class="book-cover" 
                 onerror="this.src='https://via.placeholder.com/180x280/1a1a2e/eee?text=No+Cover'"/>
            <div class="book-overlay">
                <span class="category-badge">{category}</span>
            </div>
        </div>
        <div class="book-info">
            <h3 class="book-title">{title}</h3>
            <p class="book-author">by {authors_str}</p>
            <p class="book-description">{description}</p>
            <div class="book-meta">
                <span class="meta-item">⭐ {rating}</span>
                <span class="meta-item">📅 {year}</span>
                <span class="emotion-badge" style="background: {emotion_color}20; color: {emotion_color}">
                    {emotion_icon} {emotion_score}
                </span>
            </div>
        </div>
    </div>
    '''

def recommend_books(query, category, tone):
    if not query.strip():
        return """
        <div class="empty-state">
            <div class="empty-icon">📚</div>
            <h3>Ready to discover your next read?</h3>
            <p>Describe the kind of book you're looking for above</p>
        </div>
        """
    
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    
    if len(recommendations) == 0:
        return """
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <h3>No books found</h3>
            <p>Try adjusting your search or filters</p>
        </div>
        """
    
    cards_html = "".join([create_book_card(row) for _, row in recommendations.iterrows()])
    
    return f'''
    <div class="results-header">
        <span class="results-count">Found {len(recommendations)} books</span>
    </div>
    <div class="books-grid">
        {cards_html}
    </div>
    '''

# ----------------------------
# Custom CSS
# ----------------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+Pro:wght@300;400;600&display=swap');

:root {
    --bg-primary: #0f0f1a;
    --bg-secondary: #1a1a2e;
    --bg-card: #16213e;
    --accent-gold: #d4a574;
    --accent-copper: #b8860b;
    --text-primary: #f5f5f5;
    --text-secondary: #a0a0b0;
    --text-muted: #6b7280;
    --border-color: #2a2a4a;
    --shadow: 0 10px 40px rgba(0,0,0,0.4);
}

.gradio-container {
    background: linear-gradient(135deg, var(--bg-primary) 0%, #1a1a2e 50%, #0d1b2a 100%) !important;
    min-height: 100vh;
    font-family: 'Source Sans Pro', sans-serif !important;
}

.main-header {
    text-align: center;
    padding: 3rem 1rem 2rem;
    background: linear-gradient(180deg, rgba(212,165,116,0.1) 0%, transparent 100%);
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 2rem;
}

.main-header h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 3.5rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, var(--accent-gold) 0%, #f5e6d3 50%, var(--accent-copper) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0 !important;
    letter-spacing: -0.02em;
}

.main-header p {
    color: var(--text-secondary) !important;
    font-size: 1.2rem !important;
    font-weight: 300;
    margin: 0;
}

.search-section {
    max-width: 1000px;
    margin: 0 auto 2rem;
    padding: 2rem;
    background: var(--bg-secondary);
    border-radius: 20px;
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow);
}

.search-section label {
    font-family: 'Playfair Display', serif !important;
    color: var(--accent-gold) !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

.search-section input, .search-section select {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
}

.search-section input:focus, .search-section select:focus {
    border-color: var(--accent-gold) !important;
    box-shadow: 0 0 0 3px rgba(212,165,116,0.2) !important;
}

.search-section input::placeholder {
    color: var(--text-muted) !important;
}

.search-btn {
    background: linear-gradient(135deg, var(--accent-gold) 0%, var(--accent-copper) 100%) !important;
    color: var(--bg-primary) !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    padding: 1rem 2.5rem !important;
    border-radius: 12px !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.search-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(212,165,116,0.3) !important;
}

.examples-section {
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border-color);
}

.examples-section p {
    color: var(--text-muted) !important;
    font-size: 0.9rem !important;
    margin-bottom: 0.75rem !important;
}

.example-btn {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 20px !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.85rem !important;
    margin: 0.25rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.example-btn:hover {
    background: var(--accent-gold) !important;
    color: var(--bg-primary) !important;
    border-color: var(--accent-gold) !important;
}

.books-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1.5rem;
    padding: 1rem;
    max-width: 1400px;
    margin: 0 auto;
}

.book-card {
    background: var(--bg-card);
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
    display: flex;
    flex-direction: column;
}

.book-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 20px 50px rgba(0,0,0,0.5);
    border-color: var(--accent-gold);
}

.book-cover-container {
    position: relative;
    width: 100%;
    height: 320px;
    overflow: hidden;
    background: linear-gradient(135deg, #1a1a2e, #0f0f1a);
}

.book-cover {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.5s ease;
}

.book-card:hover .book-cover {
    transform: scale(1.05);
}

.book-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    padding: 1rem;
    background: linear-gradient(180deg, rgba(0,0,0,0.7) 0%, transparent 100%);
}

.category-badge {
    background: var(--accent-gold);
    color: var(--bg-primary);
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.book-info {
    padding: 1.25rem;
    flex-grow: 1;
    display: flex;
    flex-direction: column;
}

.book-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    margin: 0 0 0.5rem 0 !important;
    line-height: 1.3 !important;
}

.book-author {
    color: var(--accent-gold) !important;
    font-size: 0.9rem !important;
    margin: 0 0 0.75rem 0 !important;
    font-weight: 400;
}

.book-description {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    line-height: 1.5 !important;
    margin: 0 0 1rem 0 !important;
    flex-grow: 1;
}

.book-meta {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
    padding-top: 0.75rem;
    border-top: 1px solid var(--border-color);
}

.meta-item {
    color: var(--text-muted);
    font-size: 0.8rem;
}

.emotion-badge {
    padding: 0.25rem 0.6rem;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 500;
}

.results-header {
    max-width: 1400px;
    margin: 0 auto 1rem;
    padding: 0 1rem;
}

.results-count {
    color: var(--text-secondary);
    font-size: 0.95rem;
    font-weight: 500;
}

.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    max-width: 500px;
    margin: 2rem auto;
}

.empty-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    opacity: 0.7;
}

.empty-state h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--text-primary) !important;
    font-size: 1.5rem !important;
    margin: 0 0 0.5rem 0 !important;
}

.empty-state p {
    color: var(--text-secondary) !important;
    font-size: 1rem !important;
    margin: 0 !important;
}

/* Fix Gradio defaults */
.dark, .gradio-container.dark {
    --body-background-fill: transparent !important;
}

footer { display: none !important; }
"""

# ----------------------------
# UI
# ----------------------------
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "😊 Happy", "😲 Surprising", "😠 Angry", "😰 Suspenseful", "😢 Sad"]

example_queries = [
    "A thrilling mystery with unexpected twists",
    "Heartwarming story about family and love",
    "Epic fantasy with magic and adventure",
    "Dark psychological thriller",
    "Inspiring true story of overcoming odds",
    "Light romantic comedy"
]

with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as dashboard:
    
    # Header
    gr.HTML("""
        <div class="main-header">
            <h1>Semantic Book Recommender</h1>
            <p>Discover your next favorite read with AI-powered recommendations</p>
        </div>
    """)
    
    # Search Section
    with gr.Group(elem_classes="search-section"):
        with gr.Row():
            user_query = gr.Textbox(
                label="What kind of book are you looking for?",
                placeholder="Describe the story, mood, or theme you're interested in...",
                lines=2,
                scale=3
            )
        
        with gr.Row():
            category_dropdown = gr.Dropdown(
                categories, 
                value="All", 
                label="Category",
                scale=1
            )
            tone_dropdown = gr.Dropdown(
                tones, 
                value="All", 
                label="Emotional Tone",
                scale=1
            )
            submit_button = gr.Button(
                "✨ Find Books", 
                elem_classes="search-btn",
                scale=1
            )
        
        # Example queries
        gr.HTML('<div class="examples-section"><p>Try these searches:</p></div>')
        with gr.Row():
            for example in example_queries[:3]:
                gr.Button(example, elem_classes="example-btn", size="sm").click(
                    lambda e=example: e, outputs=user_query
                )
        with gr.Row():
            for example in example_queries[3:]:
                gr.Button(example, elem_classes="example-btn", size="sm").click(
                    lambda e=example: e, outputs=user_query
                )
    
    # Results
    output = gr.HTML(
        value="""
        <div class="empty-state">
            <div class="empty-icon">📚</div>
            <h3>Ready to discover your next read?</h3>
            <p>Describe the kind of book you're looking for above</p>
        </div>
        """
    )
    
    # Event handlers
    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )
    
    user_query.submit(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    dashboard.launch()
