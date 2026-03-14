import os
import pdfplumber
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_pymongo import PyMongo
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# 1. IMMEDIATE BINDING: Use a simple CORS setup to speed up boot
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# 2. CONFIG
mongo_uri = os.getenv('MONGO_URI', '').strip()
if "tlsAllowInvalidCertificates" not in mongo_uri:
    mongo_uri += "&tlsAllowInvalidCertificates=true"

app.config['MONGO_URI'] = mongo_uri
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET', 'fallback-secret').strip()
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=5)

# 3. LAZY EXTENSIONS: Initialize but don't connect yet
mongo = PyMongo()
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# Important: This tells Flask-PyMongo to wait until the first request to connect
with app.app_context():
    mongo.init_app(app)

ai_model = None

def get_ai_model():
    global ai_model
    if ai_model is None:
        from sentence_transformers import SentenceTransformer
        ai_model = SentenceTransformer('all-MiniLM-L6-v2')
    return ai_model

@app.route('/')
def home():
    return "✅ Backend is Live!", 200

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
            return jsonify({
                "access_token": token, 
                "user": {"name": user['name'], "email": user['email']}
            }), 200
        return jsonify({"msg": "Invalid credentials"}), 401
    except Exception as e:
        return jsonify({"msg": str(e)}), 500

@app.route('/analyze', methods=['POST'])
@jwt_required()
def analyze():
    try:
        import nltk
        from sklearn.metrics.pairwise import cosine_similarity
        import google.generativeai as genai
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')

        model = get_ai_model()
        user_email = get_jwt_identity()
        
        if 'file' in request.files:
            file = request.files['file']
            text_content = " ".join([page.extract_text() for page in pdfplumber.open(file).pages if page.extract_text()])
        else:
            text_content = request.form.get('text', '')

        if not text_content.strip():
            return jsonify({"msg": "No content"}), 400

        sentences = nltk.sent_tokenize(" ".join(text_content.split()))
        previous_scans = list(mongo.db.scans.find({}, {"text": 1}).limit(20))
        all_prev_text = " ".join([s.get('text', '') for s in previous_scans])
        prev_sentences = nltk.sent_tokenize(all_prev_text) if all_prev_text else []

        detailed_analysis = []
        total_plagiarized_count = 0
        
        if prev_sentences:
            curr_embeddings = model.encode(sentences)
            prev_embeddings = model.encode(prev_sentences)
            sim_matrix = cosine_similarity(curr_embeddings, prev_embeddings)

            for i, s in enumerate(sentences):
                max_sim = max(sim_matrix[i]) if len(prev_sentences) > 0 else 0
                is_plag = bool(max_sim > 0.75) 
                if is_plag: total_plagiarized_count += 1
                detailed_analysis.append({"text": s, "isPlagiarized": is_plag})
        else:
            detailed_analysis = [{"text": s, "isPlagiarized": False} for s in sentences]

        score = int((total_plagiarized_count / len(detailed_analysis)) * 100) if detailed_analysis else 0

        mongo.db.scans.insert_one({
            "user_email": user_email, "text": text_content, "score": score,
            "analysis": detailed_analysis, "timestamp": datetime.utcnow()
        })
        return jsonify({"percentage": score, "analysis": detailed_analysis})
    except Exception as e:
        return jsonify({"msg": str(e)}), 500

@app.route('/history', methods=['GET', 'POST'])
@jwt_required()
def get_history():
    if request.method == 'POST': return jsonify({"msg": "ok"}), 200
    try:
        user_email = get_jwt_identity()
        scans = mongo.db.scans.find({"user_email": user_email}).sort("timestamp", -1)
        output = [{"id": str(s['_id']), "full_text": s.get('text', ''), "score": s.get('score', 0), "date": s.get('timestamp').strftime("%Y-%m-%d")} for s in scans]
        return jsonify(output), 200
    except Exception as e:
        return jsonify({"msg": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)