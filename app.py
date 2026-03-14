import os
import re
import random
import sys
import io
import pdfplumber
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_pymongo import PyMongo
from dotenv import load_dotenv
from pymongo.errors import ConnectionFailure

# --- AI & SEMANTIC IMPORTS ---
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import google.generativeai as genai

load_dotenv()

# --- INITIALIZE AI MODELS ---
print("⏳ Loading NLTK resources...")
nltk.download('punkt')
nltk.download('punkt_tab')

print("⏳ Loading Sentence Transformer (all-MiniLM-L6-v2)...")
# This loads the model. On the first run, it will download ~80MB.
ai_model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure Gemini
GEMINI_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyBiRuD5rmvmUTVecNY335pC85p3Z81Zj5E')
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
print("✅ AI Models Ready.")

# --- DATABASE CONFIG ---
mongo_uri = os.getenv('MONGO_URI', '').strip()
if not mongo_uri:
    print("⚠️ CRITICAL ERROR: MONGO_URI not found in .env file!")

app = Flask(__name__)
# Relaxed CORS for Hackathon development
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

app.config['MONGO_URI'] = mongo_uri
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET', 'fallback-secret').strip()
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=5) 

mongo = PyMongo(app)

# --- CONNECTION CHECK ---
try:
    mongo.cx.admin.command('ping')
    print("✅ MongoDB Connected Successfully!")
except Exception as e:
    print(f"❌ MongoDB Connection Failed: {e}")

bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# --- AUTH ROUTES ---
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        if mongo.db.users.find_one({"email": data.get('email')}):
            return jsonify({"msg": "User already exists"}), 400
        hashed_pw = bcrypt.generate_password_hash(data.get('password')).decode('utf-8')
        mongo.db.users.insert_one({
            "name": data.get('name'), "email": data.get('email'), "password": hashed_pw,
            "scans_count": 0, "created_at": datetime.utcnow()
        })
        return jsonify({"msg": "User created"}), 201
    except Exception as e:
        return jsonify({"msg": str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        user = mongo.db.users.find_one({"email": data.get('email')})
        if user and bcrypt.check_password_hash(user['password'], data.get('password')):
            token = create_access_token(identity=str(user['email']))
            return jsonify({"access_token": token, "user": {"name": user['name'], "email": user['email']}}), 200
        return jsonify({"msg": "Invalid credentials"}), 401
    except Exception as e:
        return jsonify({"msg": str(e)}), 500

@app.route('/profile', methods=['GET', 'PUT'])
@jwt_required()
def profile():
    user_email = get_jwt_identity()
    user = mongo.db.users.find_one({"email": user_email})
    if not user: return jsonify({"msg": "User not found"}), 404
    if request.method == 'GET':
        return jsonify({
            "name": user.get('name', 'N/A'), 
            "email": user.get('email'), 
            "scans": user.get('scans_count', 0)
        }), 200
    if request.method == 'PUT':
        data = request.json
        mongo.db.users.update_one({"email": user_email}, {"$set": {"name": data.get('name')}})
        return jsonify({"msg": "Profile updated"}), 200

# --- ANALYZE PART (SEMANTIC AI) ---
@app.route('/analyze', methods=['POST'])
@jwt_required()
def analyze():
    try:
        user_email = get_jwt_identity()
        text_content = ""

        # 1. Content Extraction
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.pdf'):
                with pdfplumber.open(file) as pdf:
                    text_content = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            else:
                text_content = file.read().decode('utf-8')
        else:
            text_content = request.form.get('text', '')

        if not text_content or not text_content.strip():
            return jsonify({"msg": "No text content found."}), 400

        text_content = " ".join(text_content.split())
        sentences = nltk.sent_tokenize(text_content)
        
        # 2. Compare against Database (Safe Retrieval)
        # We use .get('text') to avoid the KeyError 'text' if old data is buggy
        previous_scans = list(mongo.db.scans.find({}, {"text": 1}).limit(50))
        all_prev_text = " ".join([s.get('text', '') for s in previous_scans if s.get('text')])
        prev_sentences = nltk.sent_tokenize(all_prev_text) if all_prev_text else []

        detailed_analysis = []
        total_plagiarized_count = 0
        
        if prev_sentences:
            curr_embeddings = ai_model.encode(sentences)
            prev_embeddings = ai_model.encode(prev_sentences)
            sim_matrix = cosine_similarity(curr_embeddings, prev_embeddings)

            for i, s in enumerate(sentences):
                # Find best match in database
                max_sim = max(sim_matrix[i]) if len(prev_sentences) > 0 else 0
                is_plag = bool(max_sim > 0.75) 
                
                if is_plag: total_plagiarized_count += 1
                detailed_analysis.append({"text": s, "isPlagiarized": is_plag})
        else:
            # First scan in database
            detailed_analysis = [{"text": s, "isPlagiarized": False} for s in sentences]

        score = int((total_plagiarized_count / len(detailed_analysis)) * 100) if detailed_analysis else 0

        # 3. AI Insights via Gemini
        ai_insight = "Document appears highly original based on current records."
        if score > 15:
            try:
                prompt = f"Analyze this plagiarism report. Score: {score}%. Text snippet: {text_content[:600]}. Briefly explain why this might be flagged."
                response = gemini_model.generate_content(prompt)
                ai_insight = response.text
            except:
                ai_insight = "AI explanation unavailable at this moment."

        # 4. Save & Update
        mongo.db.scans.insert_one({
            "user_email": user_email,
            "text": text_content,
            "score": score,
            "analysis": detailed_analysis,
            "ai_insight": ai_insight,
            "timestamp": datetime.utcnow()
        })
        mongo.db.users.update_one({"email": user_email}, {"$inc": {"scans_count": 1}})

        return jsonify({
            "percentage": score,
            "analysis": detailed_analysis,
            "extracted_text": text_content,
            "ai_insight": ai_insight
        })
    except Exception as e:
        print(f"❌ Error in analyze: {e}")
        return jsonify({"msg": str(e)}), 500

@app.route('/history', methods=['GET'])
@jwt_required()
def get_history():
    try:
        user_email = get_jwt_identity()
        scans = mongo.db.scans.find({"user_email": user_email}).sort("timestamp", -1)
        output = []
        for s in scans:
            output.append({
                "id": str(s['_id']),
                "full_text": s.get('text', 'No content'), 
                "score": s.get('score', 0),
                "analysis": s.get('analysis', []),
                "ai_insight": s.get('ai_insight', ''),
                "date": s.get('timestamp').strftime("%Y-%m-%d") if s.get('timestamp') else "N/A"
            })
        return jsonify(output), 200
    except Exception as e:
        print(f"❌ Error in history: {e}")
        return jsonify({"msg": str(e)}), 500

if __name__ == '__main__':
    # use_reloader=False is safer when loading large AI models
    app.run(debug=True, port=5000, use_reloader=False)