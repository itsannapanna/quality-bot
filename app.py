from flask import Flask, render_template, request, session, redirect, url_for
from langchain.llms import Cohere
import cohere
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from prompt_builder import build_prompt
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Load the embedding model and FAISS index
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load Cohere LLM (or use another)
llm = Cohere(cohere_api_key="taDSt5JXKgh6LzXpu3h7QMVVdumdjFmcoyy1CifV", temperature=0)
vectorstore = FAISS.load_local("quality_vector_db", embeddings, allow_dangerous_deserialization=True)
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm  # the same LLM you’re using for querying
)

# QA chain using LangChain
qa_chain = load_qa_chain(llm, chain_type="stuff")


# Utility: Ensure session has chat storage
def get_chats():
    if "chats" not in session:
        session["chats"] = {}
    if "titles" not in session:
        session["titles"] = {}
    return session["chats"], session["titles"]

@app.route("/", methods=["GET", "POST"])
def index():
    chats, titles = get_chats()

    # Start new chat if needed
    if "active_chat" not in session or session["active_chat"] not in chats:
        new_id = str(uuid.uuid4())
        chats[new_id] = []
        titles[new_id] = "New Chat"
        session["active_chat"] = new_id
        session.modified = True

    active_id = session["active_chat"]
    history = chats.get(active_id, [])
    interpretation = ""
    main_response = ""
    user_question = ""

    if request.method == "POST":
        user_question = request.form["question"].strip()

                # Use LangChain's QA chain with retriever
        
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

        result = rag_chain.run(user_question)
        main_response = result
        interpretation = "Response generated using custom quality documents."


        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Save to session history
        history.append((user_question, main_response, timestamp))
        chats[active_id] = history
        session["chats"] = chats
        session.modified = True

        # Auto-title chat after first message
        if len(history) == 1 and user_question:
            titles[active_id] = user_question[:25] + ("..." if len(user_question) > 25 else "")
            session["titles"] = titles

    return render_template(
        "index.html",
        history=history,
        interpretation=interpretation,
        answer=main_response,
        current_question=user_question,
        chat_ids=list(chats.keys()),
        titles=titles,
        active_id=active_id
    )

@app.route("/new", methods=["POST"])
def new_chat():
    chats, titles = get_chats()
    new_id = str(uuid.uuid4())
    chats[new_id] = []
    titles[new_id] = "New Chat"
    session["active_chat"] = new_id
    session.modified = True
    return redirect(url_for("index"))

@app.route("/switch/<chat_id>")
def switch_chat(chat_id):
    chats, _ = get_chats()
    if chat_id in chats:
        session["active_chat"] = chat_id
        session.modified = True
    return redirect(url_for("index"))

@app.route("/rename/<chat_id>", methods=["POST"])
def rename_chat(chat_id):
    new_title = request.form.get("title", "").strip()
    _, titles = get_chats()
    if chat_id in titles and new_title:
        titles[chat_id] = new_title
        session["titles"] = titles
        session.modified = True
    return redirect(url_for("index"))

@app.route("/reset", methods=["POST"])
def reset_all():
    session.clear()
    return "", 204

if __name__ == "__main__":
    app.run(debug=True)
