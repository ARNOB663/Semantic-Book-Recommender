# 📚 Semantic Book Recommender

An AI-powered book recommendation system that understands natural language queries and emotional preferences. Describe the kind of book you want to read, and get personalized recommendations based on semantic similarity and emotional tone.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)
![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)

## ✨ Features

- **Semantic Search**: Find books by describing themes, moods, or storylines in natural language
- **Emotion Filtering**: Sort recommendations by emotional tone (Happy, Sad, Suspenseful, etc.)
- **Category Filtering**: Filter by Fiction, Nonfiction, Children's books
- **Beautiful UI**: Modern, dark-themed interface with book cards and cover images
- **5,000+ Books**: Curated dataset with metadata, descriptions, and emotion analysis

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Gradio |
| **Embeddings** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | ChromaDB |
| **Orchestration** | LangChain |
| **Emotion Analysis** | DistilRoBERTa (pre-computed) |
| **Classification** | BART Zero-Shot (pre-computed) |

## 📁 Project Structure

```
Semantic-Book-Recommender/
├── gradio-dashboard.py          # Main web application
├── requirements.txt             # Python dependencies
├── data/
│   └── books.csv               # Raw dataset (7K books)
├── books_cleaned.csv           # Cleaned data
├── books_with_categories.csv   # With ML-predicted categories
├── books_with_emotions.csv     # With emotion scores (final)
├── tagged_description.txt      # Descriptions for embeddings
├── sample.ipynb                # Data cleaning & vector store setup
├── text-classification.ipynb   # Category classification pipeline
├── sentiment-analysis.ipynb    # Emotion extraction pipeline
└── cover-not-found.jpg         # Fallback book cover
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Semantic-Book-Recommender.git
   cd Semantic-Book-Recommender
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
python gradio-dashboard.py
```

The app will start and display a local URL (usually `http://127.0.0.1:7860`). Open this in your browser.

> **Note**: First run may take 1-2 minutes to load embeddings into memory.

## 📖 How It Works

### 1. Data Pipeline

```
Raw Books Data → Cleaning → Category Classification → Emotion Analysis → Final Dataset
```

- **Cleaning**: Remove books with missing/short descriptions (<25 words)
- **Categories**: Zero-shot classification using `facebook/bart-large-mnli`
- **Emotions**: Extract 7 emotions using `j-hartmann/emotion-english-distilroberta-base`

### 2. Recommendation Engine

```
User Query → Embedding → Vector Similarity Search → Category Filter → Emotion Sort → Results
```

1. **Embed Query**: Convert natural language query to vector using `all-MiniLM-L6-v2`
2. **Search**: Find top 50 similar book descriptions in ChromaDB
3. **Filter**: Apply category filter if selected
4. **Sort**: Rank by selected emotional tone
5. **Return**: Top 16 book recommendations

### 3. Emotion Mapping

| UI Option | Emotion Column |
|-----------|----------------|
| 😊 Happy | `joy` |
| 😲 Surprising | `surprise` |
| 😠 Angry | `anger` |
| 😰 Suspenseful | `fear` |
| 😢 Sad | `sadness` |

## 💡 Example Queries

Try these searches to get started:

- *"A thrilling mystery with unexpected twists"*
- *"Heartwarming story about family and love"*
- *"Epic fantasy with magic and adventure"*
- *"Dark psychological thriller"*
- *"Inspiring true story of overcoming odds"*
- *"Light romantic comedy"*

## 🔧 Configuration

### Environment Variables

Create a `.env` file (optional, for OpenAI features):

```env
OPENAI_API_KEY=your_api_key_here
```

### Customization

- **Number of results**: Modify `final_top_k` in `retrieve_semantic_recommendations()`
- **Embedding model**: Change `model_name` in `HuggingFaceEmbeddings()`
- **UI theme**: Modify `custom_css` in `gradio-dashboard.py`

## 📊 Dataset

The project uses the [7K Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) dataset from Kaggle, enhanced with:

- **Category predictions** for books without categories
- **Emotion scores** (anger, disgust, fear, joy, sadness, surprise, neutral)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Dylan Castillo](https://www.kaggle.com/dylanjcastillo) for the books dataset
- [Hugging Face](https://huggingface.co/) for transformer models
- [LangChain](https://langchain.com/) for the orchestration framework
- [Gradio](https://gradio.app/) for the UI framework

