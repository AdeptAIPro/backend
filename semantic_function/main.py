# import os
# import boto3
# import numpy as np
# from sentence_transformers import SentenceTransformer, util

# # Connect to DynamoDB
# dynamodb = boto3.resource(
#     'dynamodb',
#     aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#     region_name='ap-south-1'
# )
# table = dynamodb.Table('resume_metadata')
# feedback_table = dynamodb.Table('resume_feedback')

# # Load pre-trained sentence transformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Default weight (can adapt over time)
# weight_multiplier = 1.0
# FEEDBACK_THRESHOLD = 10  # after 10 feedback entries, trigger self-training


# def nlrga_grade(score_int):
#     """NLRAG grading logic"""
#     if score_int >= 85:
#         return 'A'
#     elif score_int >= 70:
#         return 'B'
#     elif score_int >= 50:
#         return 'C'
#     else:
#         return 'D'


# def get_updated_multiplier():
#     """Learn from feedback and adjust weight multiplier."""
#     try:
#         response = feedback_table.scan()
#         feedback_items = response.get('Items', [])
#         if not feedback_items:
#             return 1.0

#         good = [item['Score'] for item in feedback_items if item['Feedback'] == 'good']
#         bad = [item['Score'] for item in feedback_items if item['Feedback'] == 'bad']

#         if not good or not bad:
#             return 1.0

#         good_avg = np.mean(good)
#         bad_avg = np.mean(bad)
#         multiplier = min(max((good_avg - bad_avg) / 20 + 1, 0.5), 1.5)
#         return multiplier
#     except Exception as e:
#         # print(f"Feedback adjustment failed: {e}")
#         return 1.0


# def retrain_embeddings():
#     """
#     Self-trainable: Update embeddings in DynamoDB when enough feedback is collected.
#     """
#     response = feedback_table.scan()
#     feedback_items = response.get('Items', [])

#     if len(feedback_items) < FEEDBACK_THRESHOLD:
#         print("Not enough feedback yet to retrain embeddings.")
#         return

#     print("Retraining embeddings based on updated resume text...")
#     response = table.scan()
#     items = response.get('Items', [])

#     for item in items:
#         resume_text = item.get('ResumeText', '')
#         skills = " ".join(item.get('Skills', []))
#         combined_text = f"{resume_text} {skills}"
#         embedding = model.encode(combined_text).tolist()

#         # Update DynamoDB (store embedding as a list of floats)
#         table.update_item(
#             Key={'email': item['email']},  # assuming email is the primary key
#             UpdateExpression='SET embedding = :e',
#             ExpressionAttributeValues={':e': embedding}
#         )
#     print("Embeddings updated successfully.")


# def semantic_search(user_input, top_k=10):
#     """
#     Semantic search + NLRAG grading + self-training feedback.
#     """
#     global weight_multiplier
#     weight_multiplier = get_updated_multiplier()

#     response = table.scan()
#     items = response.get('Items', [])

#     documents = []
#     embeddings = []
#     for item in items:
#         resume_text = item.get('ResumeText', '')
#         skills = " ".join(item.get('Skills', []))
#         combined_text = f"{resume_text} {skills}"

#         if 'embedding' in item:
#             embedding = np.array(item['embedding'])
#         else:
#             embedding = model.encode(combined_text)
#         documents.append(item)
#         embeddings.append(embedding)

#     user_embedding = model.encode(user_input)
#     cosine_scores = util.cos_sim(user_embedding, np.array(embeddings))[0]

#     top_results = sorted(
#         zip(documents, cosine_scores),
#         key=lambda x: x[1],
#         reverse=True
#     )[:top_k]

#     matching_documents = []
#     for item, score in top_results:
#         experience = item.get('Experience', 0)
#         experience_str = f"{experience} years" if experience else "0 years"

#         adjusted_score = score.item() * weight_multiplier
#         score_int = max(1, min(int(adjusted_score * 100), 100))
#         grade = nlrga_grade(score_int)

#         matching_documents.append({
#             'FullName': item.get('FullName'),
#             'email': item.get('email'),
#             'phone': item.get('phone'),
#             'Skills': item.get('Skills'),
#             'Experience': experience_str,
#             'SourceURL': item.get('SourceURL'),
#             'Score': score_int,
#             'Grade': grade
#         })

#     return matching_documents


# def record_feedback(candidate_email, score, feedback):
#     """
#     # Record user feedback into the feedback table.
#     """
#     feedback_table.put_item(Item={
#         'CandidateEmail': candidate_email,
#         'Score': score,
#         'Feedback': feedback
#     })

#     # Check if retraining is needed after feedback
#     retrain_embeddings()


# if __name__ == "__main__":
#     user_query = "we are looking for python developer with 5 years of experience in data science" 

#     results = semantic_search(user_query)
#     if results:
#         for r in results:
#             print(f"\nName: {r.get('FullName')}\nEmail: {r.get('email')}\nSkills: {r.get('Skills')}\nExperience: {r.get('Experience')}\nScore: {r.get('Score')}\nGrade (NLRAG): {r.get('Grade')}")
#             # Example feedback (in real use, collect from UI)
#             # record_feedback(r.get('email'), r.get('Score'), 'good')
#     else:
#         print("No matches found.")





# import os
# import boto3
# import numpy as np
# from sentence_transformers import SentenceTransformer, util
# from flask import Flask, request, jsonify
# from flask_cors import CORS

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)  # Enable CORS for all origins

# # Connect to DynamoDB
# dynamodb = boto3.resource(
#     'dynamodb',
#     aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#     region_name='ap-south-1'
# )
# table = dynamodb.Table('resume_metadata')
# feedback_table = dynamodb.Table('resume_feedback')

# # Load stronger sentence transformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Config
# weight_multiplier = 1.0
# FEEDBACK_THRESHOLD = 10
# MIN_SCORE_THRESHOLD =40   # skip low matches


# def nlrga_grade(score_int):
#     if score_int >= 85:
#         return 'A'
#     elif score_int >= 70:
#         return 'B'
#     elif score_int >= 50:
#         return 'C'
#     else:
#         return 'D'


# def get_updated_multiplier():
#     try:
#         response = feedback_table.scan()
#         feedback_items = response.get('Items', [])
#         if not feedback_items:
#             return 1.0

#         good = [item['Score'] for item in feedback_items if item.get('Feedback') == 'good']
#         bad = [item['Score'] for item in feedback_items if item.get('Feedback') == 'bad']

#         if not good or not bad:
#             return 1.0

#         good_avg = np.mean(good)
#         bad_avg = np.mean(bad)
#         return min(max((good_avg - bad_avg) / 20 + 1, 0.5), 1.5)
#     except Exception:
#         return 1.0


# def retrain_embeddings():
#     response = feedback_table.scan()
#     feedback_items = response.get('Items', [])

#     if len(feedback_items) < FEEDBACK_THRESHOLD:
#         print("Not enough feedback yet to retrain embeddings.")
#         return

#     print("Retraining embeddings...")
#     response = table.scan()
#     items = response.get('Items', [])

#     for item in items:
#         resume_text = item.get('ResumeText', '') or ''
#         skills = " ".join(item.get('Skills', [])) or ''
#         combined_text = f"{resume_text} {skills}".strip()
#         if not combined_text:
#             continue

#         embedding = model.encode(combined_text).tolist()
#         table.update_item(
#             Key={'email': item['email']},
#             UpdateExpression='SET embedding = :e',
#             ExpressionAttributeValues={':e': embedding}
#         )
#     print("Embeddings updated.")


# def semantic_search(user_input, top_k=10):
#     global weight_multiplier
#     weight_multiplier = get_updated_multiplier()

#     # Lowercased keyword terms from query
#     query_terms = set(user_input.lower().split())

#     response = table.scan()
#     items = response.get('Items', [])

#     documents = []
#     embeddings = []

#     for item in items:
#         fullname = item.get('FullName')
#         email = item.get('email')
#         if not fullname or not email:
#             continue  # skip incomplete records

#         resume_text = item.get('ResumeText', '') or ''
#         skills_list = item.get('Skills', []) or []
#         skills = " ".join(skills_list).strip()

#         combined_text = f"{resume_text} {skills}".lower()
#         if not combined_text:
#             continue

#         # Strict filter: only include candidates with at least one keyword match
#         if not any(term in combined_text for term in query_terms):
#             continue

#         embedding = np.array(item['embedding']) if 'embedding' in item else model.encode(combined_text)
#         documents.append(item)
#         embeddings.append(embedding)

#     if not documents:
#         return []

#     embeddings = np.vstack(embeddings)
#     user_embedding = model.encode(user_input)
#     cosine_scores = util.cos_sim(user_embedding, embeddings)[0]

#     matching_documents = []
#     for item, score in zip(documents, cosine_scores):
#         adjusted_score = score.item() * weight_multiplier
#         score_int = max(1, min(int(adjusted_score * 100), 100))

#         if score_int < MIN_SCORE_THRESHOLD:
#             continue  # skip weak matches

#         raw_experience = item.get('Experience')
#         try:
#             experience = int(raw_experience)
#         except (TypeError, ValueError):
#             experience = 0

#         matching_documents.append({
#             'FullName': fullname,
#             'email': email,
#             'phone': item.get('phone', ''),
#             'Skills': item.get('Skills', []),
#             'Experience': f"{experience} years",
#             'SourceURL': item.get('SourceURL', ''),
#             'Score': score_int,
#             'Grade': nlrga_grade(score_int)
#         })

#     matching_documents = sorted(matching_documents, key=lambda x: x['Score'], reverse=True)[:top_k]
#     return matching_documents


# def record_feedback(candidate_email, score, feedback):
#     feedback_table.put_item(Item={
#         'CandidateEmail': candidate_email,
#         'Score': score,
#         'Feedback': feedback
#     })
#     retrain_embeddings()


# @app.route('/search', methods=['POST'])
# def api_search():
#     data = request.json
#     user_input = data.get('query', '')
#     if not user_input:
#         return jsonify({'error': 'Empty query'}), 400

#     try:
#         results = semantic_search(user_input)
#         return jsonify(results)
#     except Exception as e:
#         print(f"Search error: {e}")
#         return jsonify({'error': 'Internal server error'}), 500


# @app.route('/feedback', methods=['POST'])
# def api_feedback():
#     data = request.json
#     email = data.get('email')
#     score = data.get('score')
#     feedback_value = data.get('feedback')

#     if not all([email, score, feedback_value]):
#         return jsonify({'error': 'Missing fields'}), 400

#     try:
#         record_feedback(email, score, feedback_value)
#         return jsonify({'status': 'success'})
#     except Exception as e:
#         print(f"Feedback error: {e}")
#         return jsonify({'error': 'Internal server error'}), 500


# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000)


import os
import boto3
import numpy as np
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# AWS & DynamoDB
dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name='ap-south-1'
)
table = dynamodb.Table('resume_metadata')
feedback_table = dynamodb.Table('resume_feedback')

model = SentenceTransformer('all-MiniLM-L6-v2')

# Configuration
weight_multiplier = 1.0
FEEDBACK_THRESHOLD = 10
MIN_SCORE_THRESHOLD = 40
embedding_cache = []  # Cache for embeddings + metadata


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
        feedback_items = feedback_table.scan(ProjectionExpression='Score, Feedback').get('Items', [])
        if not feedback_items:
            return 1.0

        good = [item['Score'] for item in feedback_items if item.get('Feedback') == 'good']
        bad = [item['Score'] for item in feedback_items if item.get('Feedback') == 'bad']

        if not good or not bad:
            return 1.0

        return min(max((np.mean(good) - np.mean(bad)) / 20 + 1, 0.5), 1.5)
    except Exception:
        return 1.0


def retrain_embeddings():
    feedback_items = feedback_table.scan(ProjectionExpression='CandidateEmail').get('Items', [])
    if len(feedback_items) < FEEDBACK_THRESHOLD:
        print("Not enough feedback yet to retrain embeddings.")
        return

    print("Retraining embeddings...")
    items = table.scan(ProjectionExpression='email, ResumeText, Skills').get('Items', [])

    for item in items:
        resume_text = item.get('ResumeText', '') or ''
        skills = " ".join(item.get('Skills', [])) or ''
        combined_text = f"{resume_text} {skills}".strip()
        if not combined_text:
            continue

        embedding = model.encode(combined_text, convert_to_numpy=True).tolist()
        table.update_item(
            Key={'email': item['email']},
            UpdateExpression='SET embedding = :e',
            ExpressionAttributeValues={':e': embedding}
        )
    print("Embeddings updated.")
    load_embeddings_to_memory()  # Refresh cache


def load_embeddings_to_memory():
    """Loads all resume embeddings and metadata into memory for faster access."""
    global embedding_cache
    print("Loading embeddings into memory...")

    items = table.scan(ProjectionExpression='email, FullName, phone, Skills, Experience, SourceURL, ResumeText, embedding').get('Items', [])
    cache = []
    for item in items:
        fullname = item.get('FullName')
        email = item.get('email')
        if not fullname or not email or 'embedding' not in item:
            continue

        try:
            embedding = np.array(item['embedding'])
            combined_text = f"{item.get('ResumeText', '')} {' '.join(item.get('Skills', []))}".lower()
            cache.append({
                'metadata': item,
                'embedding': embedding,
                'text': combined_text
            })
        except Exception:
            continue

    embedding_cache = cache
    print(f"Cached {len(embedding_cache)} resumes.")


def semantic_search(user_input, top_k=10):
    global weight_multiplier
    weight_multiplier = get_updated_multiplier()
    query_terms = set(user_input.lower().split())

    if not embedding_cache:
        return []

    # Pre-filter based on simple text keyword presence
    filtered = [entry for entry in embedding_cache if any(term in entry['text'] for term in query_terms)]

    if not filtered:
        return []

    all_embeddings = np.stack([entry['embedding'] for entry in filtered])
    user_embedding = model.encode(user_input, convert_to_numpy=True)

    cosine_scores = util.cos_sim(user_embedding, all_embeddings)[0].cpu().numpy()
    results = []

    for entry, score in zip(filtered, cosine_scores):
        item = entry['metadata']
        score_adj = score * weight_multiplier
        score_int = max(1, min(int(score_adj * 100), 100))
        if score_int < MIN_SCORE_THRESHOLD:
            continue

        try:
            exp = int(item.get('Experience', 0))
        except (TypeError, ValueError):
            exp = 0

        results.append({
            'FullName': item['FullName'],
            'email': item['email'],
            'phone': item.get('phone', ''),
            'Skills': item.get('Skills', []),
            'Experience': f"{exp} years",
            'SourceURL': item.get('SourceURL', ''),
            'Score': score_int,
            'Grade': nlrga_grade(score_int)
        })

    return sorted(results, key=lambda x: x['Score'], reverse=True)[:top_k]


def record_feedback(candidate_email, score, feedback):
    feedback_table.put_item(Item={
        'CandidateEmail': candidate_email,
        'Score': score,
        'Feedback': feedback
    })
    retrain_embeddings()


@app.route('/search', methods=['POST'])
def api_search():
    data = request.json
    user_input = data.get('query', '')
    if not user_input:
        return jsonify({'error': 'Empty query'}), 400

    try:
        results = semantic_search(user_input)
        return jsonify(results)
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/feedback', methods=['POST'])
def api_feedback():
    data = request.json
    email = data.get('email')
    score = data.get('score')
    feedback_value = data.get('feedback')

    if not all([email, score, feedback_value]):
        return jsonify({'error': 'Missing fields'}), 400

    try:
        record_feedback(email, score, feedback_value)
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Feedback error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == "__main__":
    load_embeddings_to_memory()  # Preload once at server start
    app.run(host='0.0.0.0', port=5000)
