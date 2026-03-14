import os
import pdfplumber
from datetime import datetime, timedelta

from flask import Flask, request, jsonify
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity,
)
from flask_pymongo import PyMongo
from dotenv import load_dotenv

import nltk
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

load_dotenv()

# -----------------------------
# NLTK SAFE LOAD
# -----------------------------
NLTK_PATH = "/opt/render/nltk_data"
os.makedirs(NLTK_PATH, exist_ok=True)
nltk.data.path.append(NLTK_PATH)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=NLTK_PATH)

# -----------------------------
# LAZY LOAD AI MODEL
# -----------------------------
ai_model = None

def get_ai_model():
    global ai_model
    if ai_model is None:
        print("Loading SentenceTransformer model...")
        from sentence_transformers import SentenceTransformer
        ai_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("AI model ready")
    return ai_model


# -----------------------------
# GEMINI SETUP
# -----------------------------
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_model = None


# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": ["https://plagirism-frontned.vercel.app"]}},
    supports_credentials=True,
)

app.config["MONGO_URI"] = os.getenv("MONGO_URI")
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=5)

mongo = PyMongo(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)


# -----------------------------
# HEALTH ROUTE (important for Render)
# -----------------------------
@app.route("/")
def health():
    return {"status": "server running"}


# -----------------------------
# DB CONNECTION TEST
# -----------------------------
try:
    mongo.cx.admin.command("ping")
    print("MongoDB connected")
except Exception as e:
    print("MongoDB error:", e)


# -----------------------------
# REGISTER
# -----------------------------
@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.json

        if mongo.db.users.find_one({"email": data.get("email")}):
            return jsonify({"msg": "User already exists"}), 400

        hashed_pw = bcrypt.generate_password_hash(data["password"]).decode("utf-8")

        mongo.db.users.insert_one({
            "name": data["name"],
            "email": data["email"],
            "password": hashed_pw,
            "scans_count": 0,
            "created_at": datetime.utcnow(),
        })

        return jsonify({"msg": "User created"}), 201

    except Exception as e:
        return jsonify({"msg": str(e)}), 500


# -----------------------------
# LOGIN
# -----------------------------
@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.json

        user = mongo.db.users.find_one({"email": data.get("email")})

        if user and bcrypt.check_password_hash(user["password"], data.get("password")):

            token = create_access_token(identity=user["email"])

            return jsonify({
                "access_token": token,
                "user": {"name": user["name"], "email": user["email"]},
            })

        return jsonify({"msg": "Invalid credentials"}), 401

    except Exception as e:
        return jsonify({"msg": str(e)}), 500


# -----------------------------
# PROFILE
# -----------------------------
@app.route("/profile", methods=["GET", "PUT"])
@jwt_required()
def profile():

    user_email = get_jwt_identity()
    user = mongo.db.users.find_one({"email": user_email})

    if not user:
        return jsonify({"msg": "User not found"}), 404

    if request.method == "GET":
        return jsonify({
            "name": user.get("name"),
            "email": user.get("email"),
            "scans": user.get("scans_count", 0),
        })

    if request.method == "PUT":

        data = request.json

        mongo.db.users.update_one(
            {"email": user_email},
            {"$set": {"name": data.get("name")}},
        )

        return jsonify({"msg": "Profile updated"})


# -----------------------------
# ANALYZE DOCUMENT
# -----------------------------
@app.route("/analyze", methods=["POST"])
@jwt_required()
def analyze():

    try:
        user_email = get_jwt_identity()
        text_content = ""

        # FILE UPLOAD
        if "file" in request.files:

            file = request.files["file"]

            if file.filename.endswith(".pdf"):

                with pdfplumber.open(file) as pdf:
                    text_content = " ".join([
                        page.extract_text()
                        for page in pdf.pages
                        if page.extract_text()
                    ])

            else:
                text_content = file.read().decode("utf-8")

        else:
            if request.is_json:
                data = request.get_json()
                text_content = data.get("text", "")
            else:
                text_content = request.form.get("text", "")

        if not text_content.strip():
            return jsonify({"msg": "No text content received"}), 400

        text_content = " ".join(text_content.split())

        sentences = nltk.sent_tokenize(text_content)

        previous_scans = list(
            mongo.db.scans.find({}, {"text": 1}).limit(50)
        )

        prev_text = " ".join(
            [s.get("text", "") for s in previous_scans if s.get("text")]
        )

        prev_sentences = nltk.sent_tokenize(prev_text) if prev_text else []

        detailed_analysis = []
        plag_count = 0

        model = get_ai_model()

        if prev_sentences:

            curr_embeddings = model.encode(sentences, convert_to_numpy=True)
            prev_embeddings = model.encode(prev_sentences, convert_to_numpy=True)

            sim_matrix = cosine_similarity(curr_embeddings, prev_embeddings)

            for i, sentence in enumerate(sentences):

                max_sim = max(sim_matrix[i])

                is_plag = max_sim > 0.75

                if is_plag:
                    plag_count += 1

                detailed_analysis.append({
                    "text": sentence,
                    "isPlagiarized": bool(is_plag),
                })

        else:
            detailed_analysis = [
                {"text": s, "isPlagiarized": False} for s in sentences
            ]

        score = int((plag_count / len(sentences)) * 100) if sentences else 0

        ai_insight = "Document appears original."

        if score > 15 and gemini_model:

            try:

                prompt = f"""
Analyze this plagiarism result.

Score: {score}%

Text snippet:
{text_content[:600]}

Explain briefly why plagiarism might be detected.
"""

                response = gemini_model.generate_content(prompt)

                ai_insight = response.text

            except Exception as e:
                print("Gemini error:", e)

        mongo.db.scans.insert_one({
            "user_email": user_email,
            "text": text_content,
            "score": score,
            "analysis": detailed_analysis,
            "ai_insight": ai_insight,
            "timestamp": datetime.utcnow(),
        })

        mongo.db.users.update_one(
            {"email": user_email},
            {"$inc": {"scans_count": 1}},
        )

        return jsonify({
            "percentage": score,
            "analysis": detailed_analysis,
            "extracted_text": text_content,
            "ai_insight": ai_insight,
        })

    except Exception as e:
        print("Analyze error:", e)
        return jsonify({"msg": str(e)}), 500


# -----------------------------
# HISTORY
# -----------------------------
@app.route("/history", methods=["GET"])
@jwt_required()
def history():

    try:

        user_email = get_jwt_identity()

        scans = mongo.db.scans.find(
            {"user_email": user_email}
        ).sort("timestamp", -1)

        output = []

        for s in scans:

            output.append({
                "id": str(s["_id"]),
                "full_text": s.get("text", ""),
                "score": s.get("score", 0),
                "analysis": s.get("analysis", []),
                "ai_insight": s.get("ai_insight", ""),
                "date": s["timestamp"].strftime("%Y-%m-%d")
                if s.get("timestamp")
                else "N/A",
            })

        return jsonify(output)

    except Exception as e:
        return jsonify({"msg": str(e)}), 500


# -----------------------------
# SERVER START
# -----------------------------
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)