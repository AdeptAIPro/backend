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
#     except Exception:
#         # Fail silently without printing error
#         return 1.0


# def retrain_embeddings():
#     """
#     Self-trainable: Update embeddings in DynamoDB when enough feedback is collected.
#     """
#     response = feedback_table.scan()
#     feedback_items = response.get('Items', [])

#     if len(feedback_items) < FEEDBACK_THRESHOLD:
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


# def generate_rag_summary(top_documents, user_query):
#     """
#     Placeholder for a RAG summary.
#     This should ideally call an LLM like GPT, but we'll simulate it here.
#     """
#     summary = f"Based on your query '{user_query}', top candidates have skills like "
#     top_skills = set()
#     for doc in top_documents:
#         top_skills.update(doc['Skills'])
#     summary += ", ".join(list(top_skills)[:5]) + "."

#     return summary


# def semantic_search(user_input, top_k=10):
#     """
#     Semantic search + NLRAG grading + RAG summary + self-training feedback.
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

#     # Generate RAG summary from top candidates
#     rag_summary = generate_rag_summary(matching_documents, user_input)

#     return matching_documents, rag_summary


# def record_feedback(candidate_email, score, feedback):
#     """
#     Record user feedback into the feedback table.
#     """
#     feedback_table.put_item(Item={
#         'CandidateEmail': candidate_email,
#         'Score': score,
#         'Feedback': feedback
#     })

#     # Check if retraining is needed after feedback
#     retrain_embeddings()


# if __name__ == "__main__":
#     user_query = "we are looking for helthcare,aws,java,cpr,pharma"

#     results, summary = semantic_search(user_query)
#     if results:
#         print("\n=== Top Matching Candidates ===")
#         for r in results:
#             print(f"\nName: {r.get('FullName')}\nEmail: {r.get('email')}\nSkills: {r.get('Skills')}\nExperience: {r.get('Experience')}\nScore: {r.get('Score')}\nGrade (NLRAG): {r.get('Grade')}")
#             # Example feedback (in real use, collect from UI)
#             # record_feedback(r.get('email'), r.get('Score'), 'good')

#         print("\n=== RAG Summary ===")
#         print(summary)
#     else:
#         print("No matches found.")
import os
import boto3
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Initialize resources and model
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

        good = [item['Score'] for item in feedback_items if item['Feedback'] == 'good']
        bad = [item['Score'] for item in feedback_items if item['Feedback'] == 'bad']

        if not good or not bad:
            return 1.0

        multiplier = min(max((np.mean(good) - np.mean(bad)) / 20 + 1, 0.5), 1.5)
        return multiplier
    except Exception:
        return 1.0


def retrain_embeddings():
    feedback_items = feedback_table.scan(ProjectionExpression='CandidateEmail').get('Items', [])
    if len(feedback_items) < FEEDBACK_THRESHOLD:
        return

    print("Retraining embeddings based on updated resume text...")
    items = table.scan(ProjectionExpression='email, ResumeText, Skills').get('Items', [])
    texts = [f"{item.get('ResumeText', '')} {' '.join(item.get('Skills', []))}" for item in items]
    embeddings = model.encode(texts, convert_to_numpy=True)

    for item, embedding in zip(items, embeddings):
        table.update_item(
            Key={'email': item['email']},
            UpdateExpression='SET embedding = :e',
            ExpressionAttributeValues={':e': embedding.tolist()}
        )
    print("Embeddings updated successfully.")


def generate_rag_summary(top_documents, user_query):
    summary = f"Based on your query '{user_query}', top candidates have skills like "
    top_skills = {skill for doc in top_documents for skill in doc['Skills']}
    return summary + ", ".join(list(top_skills)[:5]) + "."


def semantic_search(user_input, top_k=10):
    global weight_multiplier
    weight_multiplier = get_updated_multiplier()

    items = table.scan(ProjectionExpression='email, FullName, phone, Skills, Experience, SourceURL, ResumeText, embedding').get('Items', [])
    
    documents = []
    texts_to_encode = []

    embeddings = []
    for item in items:
        if 'embedding' in item:
            embeddings.append(np.array(item['embedding']))
        else:
            texts_to_encode.append(f"{item.get('ResumeText', '')} {' '.join(item.get('Skills', []))}")
            embeddings.append(None)  # Placeholder
        documents.append(item)

    if texts_to_encode:
        encoded = model.encode(texts_to_encode, convert_to_numpy=True)
        j = 0
        for i in range(len(embeddings)):
            if embeddings[i] is None:
                embeddings[i] = encoded[j]
                j += 1

    user_embedding = model.encode(user_input, convert_to_numpy=True)
    cosine_scores = util.cos_sim(user_embedding, np.stack(embeddings))[0].cpu().numpy()

    scored_documents = sorted(
        zip(documents, cosine_scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    matching_documents = []
    for item, score in scored_documents:
        adjusted_score = score * weight_multiplier
        score_int = max(1, min(int(adjusted_score * 100), 100))
        matching_documents.append({
            'FullName': item.get('FullName'),
            'email': item.get('email'),
            'phone': item.get('phone'),
            'Skills': item.get('Skills'),
            'Experience': f"{item.get('Experience', 0)} years",
            'SourceURL': item.get('SourceURL'),
            'Score': score_int,
            'Grade': nlrga_grade(score_int)
        })

    summary = generate_rag_summary(matching_documents, user_input)
    return matching_documents, summary


def record_feedback(candidate_email, score, feedback):
    feedback_table.put_item(Item={
        'CandidateEmail': candidate_email,
        'Score': score,
        'Feedback': feedback
    })
    retrain_embeddings()


if __name__ == "__main__":
    user_query = "we are looking for helthcare,aws,java,cpr,pharma"
    results, summary = semantic_search(user_query)

    if results:
        print("\n=== Top Matching Candidates ===")
        for r in results:
            print(f"\nName: {r['FullName']}\nEmail: {r['email']}\nSkills: {r['Skills']}\nExperience: {r['Experience']}\nScore: {r['Score']}\nGrade (NLRAG): {r['Grade']}")
        print("\n=== RAG Summary ===")
        print(summary)
    else:
        print("No matches found.")
