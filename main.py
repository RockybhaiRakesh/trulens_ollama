import requests
import json
import chromadb

from langchain_community.llms import Ollama

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate

from trulens.core import Feedback, TruSession
from trulens.apps.langchain import TruChain
from trulens.providers.litellm import LiteLLM

# ---------------- Ollama + Chroma Setup ----------------
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"

CHROMA_API_KEY = "ck-"
CHROMA_TENANT = ""
CHROMA_DATABASE = "demo1"

chroma_client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE
)

collection = chroma_client.get_or_create_collection(name="demo_collection")

# ---------------- Embedding Function ----------------
def get_embedding(text: str):
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": "nomic-embed-text", "input": text}
    )
    data = response.json()
    return data["embeddings"][0]

# ---------------- Add Documents ----------------
documents = {
    "uw_info": "The University of Washington, founded in 1861 in Seattle, is a public research university with over 45,000 students.",
    "wsu_info": "Washington State University, founded in 1890, is a public research university in Pullman, Washington.",
    "starbucks_info": "Starbucks is an American multinational coffeehouse chain headquartered in Seattle, Washington."
}

for doc_id, text in documents.items():
    emb = get_embedding(text)
    collection.add(documents=[text], ids=[doc_id], embeddings=[emb])

print("✅ Documents stored in Chroma Cloud")

# ---------------- LangChain + Ollama ----------------
ollama = Ollama(base_url="http://localhost:11434", model="llama2")

full_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template="Context: {context}\n\nQuestion: {prompt}\nAnswer:",
        input_variables=["context", "prompt"],
    )
)
chat_prompt_template = ChatPromptTemplate.from_messages([full_prompt])
chain = LLMChain(llm=ollama, prompt=chat_prompt_template, verbose=True)

# ---------------- TruLens Setup ----------------
session = TruSession()
session.reset_database()

ollama_provider = LiteLLM(
    model_engine="ollama/llama2",
    api_base="http://localhost:11434"
)

relevance = Feedback(
    ollama_provider.relevance_with_cot_reasons
).on_input_output()

tru_recorder = TruChain(
    chain, app_name="Llama2_Chroma_RAG", feedbacks=[relevance]
)

# ---------------- Query Function ----------------
def rag_query(query: str):
    q_emb = get_embedding(query)
    results = collection.query(query_embeddings=[q_emb], n_results=2)
    context = " ".join([doc for docs in results["documents"] for doc in docs])
    with tru_recorder as recording:
        response = chain.invoke({"context": context, "prompt": query})
    return response["text"]

# ---------------- Run Example ----------------
if __name__ == "__main__":
    queries = [
        "What is the University of Washington?",
        "Where is Starbucks headquartered?",
        "Tell me about Washington State University."
    ]

    for q in queries:
        print(f"\n❓ Query: {q}")
        answer = rag_query(q)
        print(f"🤖 Llama2: {answer}")

    # ---------------- Run TruLens Dashboard ----------------
    from trulens.dashboard import run_dashboard
    run_dashboard(session)  # 👉 open http://localhost:8501 in browser

