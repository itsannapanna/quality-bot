from flask import Flask, render_template, request, session, redirect, url_for
import cohere
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from prompt_builder import build_prompt

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY"))

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

        # Context for LLM prompt
        conversation_context = "\n".join([
            f"User: {q}\nBot: {a}" for q, a, _ in history
        ])

        prompt = build_prompt(user_question=user_question, context=conversation_context)

        # Generate response from Cohere
        response = co.generate(
            model="command-r-plus",
            prompt=prompt,
            max_tokens=400,
            temperature=0.6
        )

        full_output = response.generations[0].text.strip()

        # Parse interpretation and answer
        if "Interpretation:" in full_output and "Response:" in full_output:
            interpretation = full_output.split("Interpretation:")[1].split("Response:")[0].strip()
            main_response = full_output.split("Response:")[1].strip()
        else:
            main_response = full_output

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
