📘 README.md – Enterprise RAG with Llama2 + Chroma Cloud + TruLens
# 🚀 Enterprise RAG System – Llama2 + Chroma Cloud + TruLens

This project demonstrates an **enterprise-level Retrieval Augmented Generation (RAG) system** with:

- **Ollama** → Local inference with Llama2 & embeddings  
- **Chroma Cloud** → Managed vector database  
- **LangChain** → Query orchestration  
- **TruLens** → Evaluation, monitoring & dashboard  

---

## 🛠️ Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running locally
- Chroma Cloud account + API Key
- Streamlit (auto-installed with TruLens) for dashboard

---

## 📦 Installation

Clone this repo and install dependencies:

```bash
pip install requests chromadb langchain-community trulens trulens-apps-langchain trulens-providers-litellm

⚙️ Setup Ollama

Make sure Ollama is running.
Pull the required models:

ollama pull llama2
ollama pull nomic-embed-text


Verify installed models:

ollama list


Expected output:

NAME                       ID              SIZE      MODIFIED
llama2:latest              78e26419b446    3.8 GB    47 minutes ago
nomic-embed-text:latest    0a109f422b47    274 MB    47 minutes ago

🔑 Configure Chroma Cloud

Replace these values in main.py:

CHROMA_API_KEY = "your_api_key"
CHROMA_TENANT = "your_tenant_id"
CHROMA_DATABASE = "your_database"

📂 How It Works

Embed & Store Documents

Uses nomic-embed-text model from Ollama

Stores embeddings in Chroma Cloud

Query Flow

User query → embedding

Chroma Cloud returns top documents

Context + query sent to Llama2 model via Ollama

Generates answer

Evaluation & Monitoring

TruLens tracks inputs, context, outputs

Feedback function checks relevance

Dashboard displays insights

▶️ Run the Project

Start the script:

python main.py


Example output:

✅ Documents stored in Chroma Cloud

❓ Query: Where is Starbucks headquartered?
🤖 Llama2: Starbucks is headquartered in Seattle, Washington.

📊 Run the Dashboard

After script execution, TruLens dashboard runs automatically.

Open in your browser:

👉 http://localhost:8501

Dashboard Features:

Queries & Responses log

Retrieved context inspection

Relevance & feedback metrics

Audit-friendly UI for managers