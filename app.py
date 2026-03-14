import os
import re
import pdfplumber
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_pymongo import PyMongo
from dotenv import load_dotenv

# --- AI & SEMANTIC IMPORTS ---
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import google.generativeai as genai

load_dotenv()

# --- INITIALIZE AI MODELS ---
print("⏳ Preparing AI Environment...")

# USE A LIGHTER MODEL FOR FAST DEPLOYMENT
print("⏳ Loading Light Semantic Model...")
ai_model = SentenceTransformer('paraphrase-albert-small-v2')

# --- ADDED YOUR API KEY HERE ---
GEMINI_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyBiRuD5rmvmUTVecNY335pC85p3Z81Zj5E')
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
print("✅ AI Models Ready.")

app = Flask(__name__)

# --- UPDATED CORS FOR VERCEL ---
CORS(app, resources={r"/*": {"origins": ["https://plagirism-frontned.vercel.app"]}}, supports_credentials=True)

app.config['MONGO_URI'] = os.getenv('MONGO_URI', '').strip()
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET', 'fallback-secret').strip()
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=5) 

mongo = PyMongo(app)

bcrypt = Bcrypt(app)
jwt = JWTManager(app)

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

@app.route('/analyze', methods=['POST'])
@jwt_required()
def analyze():
    try:
        user_email = get_jwt_identity()
        text_content = ""

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
        
        previous_scans = list(mongo.db.scans.find({}, {"text": 1}).limit(20))
        all_prev_text = " ".join([s.get('text', '') for s in previous_scans if s.get('text')])
        prev_sentences = nltk.sent_tokenize(all_prev_text) if all_prev_text else []

        detailed_analysis = []
        total_plagiarized_count = 0
        
        if prev_sentences:
            curr_embeddings = ai_model.encode(sentences)
            prev_embeddings = ai_model.encode(prev_sentences)
            sim_matrix = cosine_similarity(curr_embeddings, prev_embeddings)

            for i, s in enumerate(sentences):
                max_sim = max(sim_matrix[i]) if len(prev_sentences) > 0 else 0
                is_plag = bool(max_sim > 0.75) 
                if is_plag: total_plagiarized_count += 1
                detailed_analysis.append({"text": s, "isPlagiarized": is_plag})
        else:
            detailed_analysis = [{"text": s, "isPlagiarized": False} for s in sentences]

        score = int((total_plagiarized_count / len(detailed_analysis)) * 100) if detailed_analysis else 0

        ai_insight = "Document appears highly original."
        if score > 10:
            try:
                prompt = f"Analyze this report. Score: {score}%. Text: {text_content[:300]}. Explain briefly."
                response = gemini_model.generate_content(prompt)
                ai_insight = response.text
            except:
                ai_insight = "AI insight currently unavailable."

        mongo.db.scans.insert_one({
            "user_email": user_email, "text": text_content, "score": score,
            "analysis": detailed_analysis, "ai_insight": ai_insight, "timestamp": datetime.utcnow()
        })
        mongo.db.users.update_one({"email": user_email}, {"$inc": {"scans_count": 1}})

        return jsonify({"percentage": score, "analysis": detailed_analysis, "extracted_text": text_content, "ai_insight": ai_insight})
    except Exception as e:
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
                "id": str(s['_id']), "full_text": s.get('text', 'No content'), 
                "score": s.get('score', 0), "analysis": s.get('analysis', []),
                "ai_insight": s.get('ai_insight', ''),
                "date": s.get('timestamp').strftime("%Y-%m-%d") if s.get('timestamp') else "N/A"
            })
        return jsonify(output), 200
    except Exception as e:
        return jsonify({"msg": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)