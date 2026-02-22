import os
import sys
from datetime import datetime, timezone
from urllib.parse import quote_plus

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

def safe_print(text: str) -> None:
    """Print text safely on terminals with limited encodings (e.g., cp1252 on Windows)."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoded = text.encode(sys.stdout.encoding or "utf-8", errors="replace")
        print(encoded.decode(sys.stdout.encoding or "utf-8", errors="replace"))


def build_mongo_uri() -> str:
    direct_url = os.getenv("MONGO_DB_URL")
    if direct_url:
        return direct_url

    user = os.getenv("MONGO_USER")
    password = os.getenv("MONGO_PASSWORD")
    if not user or not password:
        raise ValueError(
            "Provide MONGO_DB_URL or both MONGO_USER and MONGO_PASSWORD in .env"
        )

    host = os.getenv("MONGO_HOST", "cluster0.fnotetk.mongodb.net")
    return (
        f"mongodb+srv://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}/?retryWrites=true&w=majority"
    )


def get_response_text(response) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    return str(content)


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing in .env")

mongo_uri = build_mongo_uri()
db_name = os.getenv("MONGO_DB_NAME", "chatbot")
collection_name = os.getenv("MONGO_COLLECTION", "users")

client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000, connectTimeoutMS=10000)
db = client[db_name]
collection = db[collection_name]

app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    question: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

try:
    client.admin.command("ping")
    safe_print(f"MongoDB connected! db={db_name}, collection={collection_name}")
except PyMongoError as err:
    raise RuntimeError(f"MongoDB connection failed: {err}") from err

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a experts in mentoring, respond to the user's question in a helpful and concise manner. If you don't know the answer, say you don't know."),
        ("placeholder", "{history}"),
        ("user", "{question}"),
    ]
)

llm = ChatGroq(api_key=groq_api_key, model="openai/gpt-oss-20b")
chain = prompt | llm

# user_id = "user123"

def get_history(user_id:str):
    chats = collection.find({"user_id" : user_id}).sort("timestamp", 1)
    history = []

    for chat in chats:
        # history.append({chat["role"], chat["message"]})
        role = chat.get("role")
        message = chat.get("message", "")
        if role == "user":
            history.append(HumanMessage(content=message))
        elif role == "assistant":
            history.append(AIMessage(content=message))

    return history

@app.get("/")
def home():
    return {"message": "Welcome to the student mentor chatbot API"}

@app.post("/chat")
def chat(request: ChatRequest):
    history = get_history(request.user_id)
    response = chain.invoke({"history": history,"question": request.question})
    answer_text = get_response_text(response)
    safe_print(f"Assistant: {answer_text}")

    docs = [
        {
            "user_id": request.user_id,
            "role": "user",
            "message": request.question,
            "timestamp": datetime.now(timezone.utc),
        },
        {
            "user_id": request.user_id,
            "role": "assistant",
            "message": answer_text,
            "timestamp": datetime.now(timezone.utc),
        },
    ]

    try:
        result = collection.insert_many(docs, ordered=True)
        safe_print(f"Saved to MongoDB: {len(result.inserted_ids)} documents")
    except PyMongoError as err:
        safe_print(f"MongoDB insert failed ({type(err).__name__}): {err}")

    return {"response" : answer_text}

# ------------------------------------------ default message store in DB
# while True:
#     question = input("Ask a question: ").strip()
#     if question.lower() in {"exit", "quit"}:
#         break

#     if not question:
#         safe_print("Please enter a non-empty question.")
#         continue

#     history = get_history(user_id)
#     response = chain.invoke({"history": history,"question": question})
#     answer_text = get_response_text(response)
#     safe_print(f"Assistant: {answer_text}")

#     docs = [
#         {
#             "user_id": user_id,
#             "role": "user",
#             "message": question,
#             "timestamp": datetime.now(timezone.utc),
#         },
#         {
#             "user_id": user_id,
#             "role": "assistant",
#             "message": answer_text,
#             "timestamp": datetime.now(timezone.utc),
#         },
#     ]

#     try:
#         result = collection.insert_many(docs, ordered=True)
#         safe_print(f"Saved to MongoDB: {len(result.inserted_ids)} documents")
#     except PyMongoError as err:
#         safe_print(f"MongoDB insert failed ({type(err).__name__}): {err}")
