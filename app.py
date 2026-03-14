import os
import pdfplumber
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_pymongo import PyMongo
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

app = Flask(__name__)

# --- CORS CONFIGURATION ---
# Allows connection from Vercel and local development ports
CORS(app, resources={r"/*": {
    "origins": [
        "https://plagirism-frontned.vercel.app", 
        "http://localhost:3000", 
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]
}}, supports_credentials=True)

# --- APP CONFIG ---
# Added &tlsAllowInvalidCertificates=true to prevent local DNS/SSL timeouts
mongo_uri = os.getenv('MONGO_URI', '').strip()
if "tlsAllowInvalidCertificates" not in mongo_uri:
    mongo_uri += "&tlsAllowInvalidCertificates=true"

app.config['MONGO_URI'] = mongo_uri
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET', 'fallback-secret').strip()
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=5) 

# Initialize Flask Extensions
mongo = PyMongo(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# Global model variable for lazy loading
ai_model = None

# --- UTILITY FUNCTIONS ---

def get_ai_model():
    """Loads heavy AI libraries only when the first scan is requested."""
    global ai_model
    if ai_model is None:
        print("⏳ First scan detected: Loading Semantic Model...")
        from sentence_transformers import SentenceTransformer
        ai_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ AI Engine Ready.")
    return ai_model

# --- ROUTES ---

@app.route('/')
def home():
    return "✅ Plagiarism Backend is Live and Fast!", 200

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        if mongo.db.users.find_one({"email": data.get('email')}):
            return jsonify({"msg": "User already exists"}), 400
        hashed_pw = bcrypt.generate_password_hash(data.get('password')).decode('utf-8')
        mongo.db.users.insert_one({
            "name": data.get('name'), 
            "email": data.get('email'), 
            "password": hashed_pw,
            "scans_count": 0, 
            "created_at": datetime.utcnow()
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
        # Move heavy imports inside to ensure fast app startup
        import nltk
        from sklearn.metrics.pairwise import cosine_similarity
        import google.generativeai as genai
        
        # Ensure NLTK resources are available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')

        model = get_ai_model()
        user_email = get_jwt_identity()
        text_content = ""

        # 1. Extraction
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
        
        # 2. Comparison
        previous_scans = list(mongo.db.scans.find({}, {"text": 1}).limit(20))
        all_prev_text = " ".join([s.get('text', '') for s in previous_scans if s.get('text')])
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

        # 3. Gemini Insights
        ai_insight = "Highly original content."
        try:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY', 'AIzaSyBiRuD5rmvmUTVecNY335pC85p3Z81Zj5E'))
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"Briefly analyze this plagiarism report. Score: {score}%. Content snippet: {text_content[:200]}"
            response = gemini_model.generate_content(prompt)
            ai_insight = response.text
        except:
            ai_insight = "Gemini insight unavailable."

        # 4. Save to DB
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
            "ai_insight": ai_insight,
            "extracted_text": text_content
        })
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
                "id": str(s['_id']), 
                "full_text": s.get('text', 'No content'), 
                "score": s.get('score', 0), 
                "analysis": s.get('analysis', []),
                "date": s.get('timestamp').strftime("%Y-%m-%d") if s.get('timestamp') else "N/A"
            })
        return jsonify(output), 200
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

if __name__ == '__main__':
    # Use environment PORT or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)