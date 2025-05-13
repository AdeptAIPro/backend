import os
from dotenv import load_dotenv
import boto3
import numpy as np
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify, render_template
import requests
from datetime import datetime, timedelta
from flask_cors import CORS
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:8081",
            "http://127.0.0.1:8081",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5055",
            "http://127.0.0.1:5055",
            "https://adeptaipro.com",
            "https://seagreen-hedgehog-452490.hostingersite.com/"
        ],
        "supports_credentials": True,
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "methods": ["GET", "POST", "OPTIONS"]
    }
})

# Connect to DynamoDB
dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name='ap-south-1'
)
table = dynamodb.Table('resume_metadata')
feedback_table = dynamodb.Table('resume_feedback')

model = SentenceTransformer('all-MiniLM-L6-v2')

weight_multiplier = 1.0
FEEDBACK_THRESHOLD = 10

# --- CEIPAL Auth and API Classes ---
class CeipalAuth:
    def __init__(self):
        self.auth_url = "https://api.ceipal.com/v1/createAuthtoken/"
        self.email = os.getenv("CEIPAL_EMAIL")
        self.password = os.getenv("CEIPAL_PASSWORD")
        self.api_key = os.getenv("CEIPAL_API_KEY")
        self.token = None
        self.token_expiry = None

    def authenticate(self) -> bool:
        if not all([self.email, self.password, self.api_key]):
            logger.error("Missing CEIPAL credentials in environment variables")
            return False

        payload = {
            "email": self.email,
            "password": self.password,
            "api_key": self.api_key,
            "json": "1"
        }
        try:
            logger.info("Attempting CEIPAL authentication...")
            response = requests.post(self.auth_url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "access_token" in data:
                self.token = data.get("access_token")
                self.token_expiry = datetime.now() + timedelta(hours=1)
                logger.info("CEIPAL authentication successful")
                return True
            else:
                error_msg = data.get("error", "Unknown error")
                logger.error(f"CEIPAL authentication failed: {error_msg}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"CEIPAL authentication request failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"CEIPAL authentication error: {str(e)}")
            return False

    def get_token(self) -> str:
        if not self.token or datetime.now() >= self.token_expiry:
            if not self.authenticate():
                raise Exception("Failed to authenticate with CEIPAL")
        return self.token

class CeipalJobPostingsAPI:
    def __init__(self, auth: CeipalAuth):
        self.auth = auth
        self.base_url = "https://api.ceipal.com"
        self.job_postings_endpoint = "/getCustomJobPostingDetails/Z3RkUkt2OXZJVld2MjFpOVRSTXoxZz09/e6e04af381e7f42eeb7f942c8bf5ab6d"
        self.job_details_endpoint = "/v1/getJobPostingDetails/"

    def get_job_postings(self, paging_length: int = 20):
        token = self.auth.get_token()
        if not token:
            raise Exception("Authentication failed")
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        params = {"paging_length": paging_length}
        
        try:
            logger.info("Fetching job postings from CEIPAL...")
            response = requests.get(
                f"{self.base_url}{self.job_postings_endpoint}",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            job_data = response.json()
            logger.info(f"Successfully fetched {len(job_data.get('results', []))} jobs")
            return job_data.get("results", [])
        except Exception as e:
            logger.error(f"Error fetching job postings: {str(e)}")
            raise

    def get_job_details(self, job_code: str):
        token = self.auth.get_token()
        if not token:
            raise Exception("Authentication failed")
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        params = {"job_id": job_code}
        
        try:
            logger.info(f"Fetching job details for job code: {job_code}")
            response = requests.get(
                f"{self.base_url}{self.job_details_endpoint}",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            job_data = response.json()
            logger.info("Successfully fetched job details")
            return job_data
        except Exception as e:
            logger.error(f"Error fetching job details: {str(e)}")
            raise

def nlrga_grade(score_int):
    if score_int >= 85:
        return 'A'
    elif score_int >= 70:
        return 'B'
    elif score_int >= 50:
        return 'C'
    else:
        return 'D'

def get_updated_multiplier():
    try:
        response = feedback_table.scan()
        feedback_items = response.get('Items', [])
        if not feedback_items:
            return 1.0
        good = [item['Score'] for item in feedback_items if item.get('Feedback') == 'good']
        bad = [item['Score'] for item in feedback_items if item.get('Feedback') == 'bad']
        if not good or not bad:
            return 1.0
        good_avg = np.mean(good)
        bad_avg = np.mean(bad)
        return min(max((good_avg - bad_avg) / 20 + 1, 0.5), 1.5)
    except Exception:
        return 1.0

def retrain_embeddings():
    feedback_items = feedback_table.scan().get('Items', [])
    if len(feedback_items) < FEEDBACK_THRESHOLD:
        return

    paginator = table.scan()
    items = []
    while True:
        items.extend(paginator.get('Items', []))
        if 'LastEvaluatedKey' in paginator:
            paginator = table.scan(ExclusiveStartKey=paginator['LastEvaluatedKey'])
        else:
            break

    for item in items:
        resume_text = item.get('ResumeText', '')
        skills = " ".join(item.get('Skills', []))
        combined_text = f"{resume_text} {skills}".strip()
        if combined_text:
            embedding = model.encode(combined_text).tolist()
            table.update_item(
                Key={'email': item['email']},
                UpdateExpression='SET embedding = :e',
                ExpressionAttributeValues={':e': embedding}
            )

def semantic_search(user_input, top_k=10):
    global weight_multiplier
    weight_multiplier = get_updated_multiplier()
    user_embedding = model.encode(user_input)

    query_terms = set(user_input.lower().split())

    paginator = table.scan()
    documents = []
    embeddings = []

    while True:
        items = paginator.get('Items', [])
        for item in items:
            resume_text = item.get('ResumeText', '') or ''
            skills = " ".join(item.get('Skills', [])).lower()
            combined_text = f"{resume_text} {skills}".lower()
            if not any(term in combined_text for term in query_terms):
                continue  # skip early if no keyword match

            if 'embedding' in item:
                embedding = np.array(item['embedding'])
            else:
                embedding = model.encode(combined_text)

            documents.append(item)
            embeddings.append(embedding)

        if 'LastEvaluatedKey' in paginator:
            paginator = table.scan(ExclusiveStartKey=paginator['LastEvaluatedKey'])
        else:
            break

    if not documents:
        return [], "No candidates matched the query."

    embeddings = np.vstack(embeddings)
    cosine_scores = util.cos_sim(user_embedding, embeddings)[0]

    top_results = sorted(
        zip(documents, cosine_scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    matching_documents = []
    for item, score in top_results:
        experience = item.get('Experience') or 0
        try:
            experience_int = int(experience)
        except (TypeError, ValueError):
            experience_int = 0

        adjusted_score = score.item() * weight_multiplier
        score_int = max(1, min(int(adjusted_score * 100), 100))
        grade = nlrga_grade(score_int)

        matching_documents.append({
            'FullName': item.get('FullName') or '',
            'email': item.get('email') or '',
            'phone': item.get('phone') or '',
            'Skills': item.get('Skills') or [],
            'Experience': f"{experience_int} years",
            'SourceURL': item.get('SourceURL') or '',
            'Score': score_int,
            'Grade': grade
        })

    summary = f"Top {len(matching_documents)} candidates matching query '{user_input}'."
    return matching_documents, summary

def keyword_search(user_input, top_k=10):
    user_keywords = set(user_input.lower().split())
    paginator = table.scan()
    scored_items = []

    while True:
        items = paginator.get('Items', [])
        for item in items:
            resume_text = (item.get('ResumeText') or '').lower()
            skills = " ".join(item.get('Skills') or []).lower()
            combined_text = f"{resume_text} {skills}"

            match_count = sum(1 for word in user_keywords if word in combined_text)
            if match_count == 0:
                continue

            score_int = min(match_count * 10, 100)
            grade = nlrga_grade(score_int)

            scored_items.append({
                'FullName': item.get('FullName') or '',
                'email': item.get('email') or '',
                'phone': item.get('phone') or '',
                'Skills': item.get('Skills') or [],
                'Experience': f"{item.get('Experience') or 0} years",
                'SourceURL': item.get('SourceURL') or '',
                'Score': score_int,
                'Grade': grade
            })

        if 'LastEvaluatedKey' in paginator:
            paginator = table.scan(ExclusiveStartKey=paginator['LastEvaluatedKey'])
        else:
            break

    sorted_items = sorted(scored_items, key=lambda x: x['Score'], reverse=True)[:top_k]
    summary = f"Keyword search found top {len(sorted_items)} candidates for query '{user_input}'."
    return sorted_items, summary

def record_feedback(candidate_email, score, feedback):
    feedback_table.put_item(Item={
        'CandidateEmail': candidate_email,
        'Score': score,
        'Feedback': feedback
    })
    retrain_embeddings()

# Update the home route to serve index.html
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        query = data.get('query', '')
        page = data.get('page', 1)
        page_size = data.get('pageSize', 10)

        if not query:
            return jsonify({'error': 'Empty query'}), 400

        logger.info(f"Processing search request: query='{query}', page={page}, page_size={page_size}")
        
        # Perform the search
        results, summary = semantic_search(query)
        
        # Calculate pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_results = results[start_idx:end_idx]

        # Ensure each candidate has all required fields for the frontend
        for candidate in paginated_results:
            candidate.setdefault('FullName', '')
            candidate.setdefault('email', '')
            candidate.setdefault('phone', '')
            candidate.setdefault('Skills', [])
            candidate.setdefault('Experience', '')
            candidate.setdefault('SourceURL', '')
            candidate.setdefault('Score', 0)
            candidate.setdefault('Grade', '')

        response_data = {
            'results': paginated_results,
            'summary': summary,
            'total': len(results),
            'page': page,
            'pageSize': page_size,
            'hasMore': end_idx < len(results)
        }

        logger.info(f"Search completed: found {len(results)} results, returning {len(paginated_results)}")
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    record_feedback(
        data.get('email'),
        data.get('score'),
        data.get('feedback')
    )
    return jsonify({'status': 'success'})

# --- CEIPAL Endpoints ---
@app.route('/api/v1/ceipal/jobs', methods=['GET'])
def get_ceipal_jobs():
    try:
        auth = CeipalAuth()
        api = CeipalJobPostingsAPI(auth)
        jobs = api.get_job_postings()
        return jsonify({"jobs": jobs})
    except Exception as e:
        logger.error(f"Error in get_ceipal_jobs endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/ceipal/getJobDetails', methods=['GET'])
def get_job_details():
    job_code = request.args.get('job_code')
    if not job_code:
        logger.error("Missing job_code parameter")
        return jsonify({"error": "Missing job_code parameter"}), 400
    
    try:
        logger.info(f"Fetching job details for job code: {job_code}")
        auth = CeipalAuth()
        if not auth.authenticate():
            logger.error("Failed to authenticate with CEIPAL")
            return jsonify({"error": "Failed to authenticate with CEIPAL"}), 401

        api = CeipalJobPostingsAPI(auth)
        job_details = api.get_job_details(job_code)
        
        if not job_details:
            logger.error(f"No job details found for job code: {job_code}")
            return jsonify({"error": "No job details found"}), 404

        logger.info(f"Successfully fetched job details for job code: {job_code}")
        return jsonify({"job_details": job_details})
    except Exception as e:
        logger.error(f"Error in get_job_details endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test_connection():
    return jsonify({"status": "ok", "message": "Backend is running"})

if __name__ == '__main__':
    # Check if CEIPAL credentials are set
    required_env_vars = ["CEIPAL_EMAIL", "CEIPAL_PASSWORD", "CEIPAL_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set the required environment variables before running the application")
    else:
        logger.info("Starting Flask application...")
        app.run(host='127.0.0.1', port=5055, debug=True)
