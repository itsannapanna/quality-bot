from flask import Flask, render_template, request, session, redirect, url_for, flash
from langchain_community.llms import Cohere
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from datetime import datetime
import cohere
import os
import uuid
from werkzeug.utils import secure_filename

# Custom utility
from prompt_builder import build_prompt

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = "data"
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# LLM and vector setup
cohere_api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(cohere_api_key)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("quality_vector_db", embeddings, allow_dangerous_deserialization=True)
llm = Cohere(cohere_api_key=cohere_api_key, temperature=0)

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm
)
qa_chain = load_qa_chain(llm, chain_type="stuff")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_chats():
    if "chats" not in session:
        session["chats"] = {}
    if "titles" not in session:
        session["titles"] = {}
    return session["chats"], session["titles"]

@app.route("/", methods=["GET", "POST"])
def index():
    chats, titles = get_chats()

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

    if request.method == "POST" and "question" in request.form:
        user_question = request.form["question"].strip()

        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

        result = rag_chain.run(user_question)
        main_response = result
        interpretation = "Response generated using custom quality documents."

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        history.append((user_question, main_response, timestamp))
        chats[active_id] = history
        session["chats"] = chats
        session.modified = True

        if len(history) == 1 and user_question:
            titles[active_id] = user_question[:25] + ("..." if len(user_question) > 25 else "")
            session["titles"] = titles

    messages = list(flash._get_flashed_messages()) if hasattr(flash, '_get_flashed_messages') else []

    return render_template(
        "index.html",
        history=history,
        interpretation=interpretation,
        answer=main_response,
        current_question=user_question,
        chat_ids=list(chats.keys()),
        titles=titles,
        active_id=active_id,
        messages=messages
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

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if 'pdf' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['pdf']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        loader = PyPDFLoader(file_path)
        pages = loader.load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200).split_documents(pages)
        vectorstore.add_documents(chunks)

        flash(f"Successfully uploaded and indexed {filename}")
        return redirect(url_for("index"))

    flash("Invalid file type")
    return redirect(url_for("index"))

@app.route("/reset", methods=["POST"])
def reset_all():
    session.clear()
    return "", 204

if __name__ == "__main__":
    app.run(debug=True)
