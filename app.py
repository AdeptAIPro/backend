import os
import threading
from datetime import datetime, timedelta
import uuid
import hmac
import hashlib
import base64
import logging
import re

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import boto3
from botocore.exceptions import ClientError
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from jose import jwt
import stripe
from functools import wraps, lru_cache

# ------------------------------------------------------------------------------
# Load environment & configure logging
# ------------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Flask setup & CORS
# ------------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {
    "origins": [
        "http://localhost:5055",
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://adeptaipro.com",
        "https://seagreen-hedgehog-452490.hostingersite.com/",
        "http://127.0.0.1:5055"
    ],
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
}})

# ------------------------------------------------------------------------------
# Stripe configuration
# ------------------------------------------------------------------------------
STRIPE_SECRET_KEY = os.getenv('STRIPE_LIVE_SECRET_KEY')
STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_LIVE_WEBHOOK_SECRET')
stripe.api_key = STRIPE_SECRET_KEY
logger.info("Running in Stripe LIVE mode")

STRIPE_PLANS = {
    'pro': {
        'monthly': {
            'price_id': 'price_1RHnLzKVj190YQJbKhavdyiI',
            'amount': 4900
        },
        'yearly': {
            'price_id': 'price_1RHnSHKVj190YQJbW4kNxO5b',
            'amount': 49900
        }
    },
    'business': {
        'monthly': {
            'price_id': 'price_1RHnPkKVj190YQJbKtOCaZgj',
            'amount': 19900
        },
        'yearly': {
            'price_id': 'price_1RHnSuKVj190YQJbOjzAdSvD',
            'amount': 119000
        }
    },
    'test': {
        'monthly': {
            'price_id': 'price_1RRzkoKVj190YQJb7LpH5mUr',
            'amount': 100
        }
    }
}

# ------------------------------------------------------------------------------
# AWS Cognito, S3 & DynamoDB clients
# ------------------------------------------------------------------------------
COGNITO_REGION = os.getenv('COGNITO_REGION')
USER_POOL_ID = os.getenv('USER_POOL_ID')
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')

cognito_client = boto3.client('cognito-idp', region_name=COGNITO_REGION)
s3_client = boto3.client('s3', region_name=COGNITO_REGION)
dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name='ap-south-1'
)

resume_table = dynamodb.Table('resume_metadata')
feedback_table = dynamodb.Table('resume_feedback')
user_details_table = dynamodb.Table('user_details')

S3_BUCKET = 'resume-bucket-adept-ai-pro'

# ------------------------------------------------------------------------------
# Cognito helper functions
# ------------------------------------------------------------------------------
def get_secret_hash(username):
    message = username + CLIENT_ID
    dig = hmac.new(
        CLIENT_SECRET.encode('utf-8'),
        msg=message.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    return base64.b64encode(dig).decode()

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            keys_url = f'https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{USER_POOL_ID}/.well-known/jwks.json'
            jwks = requests.get(keys_url).json()['keys']
            unverified_header = jwt.get_unverified_header(token)
            rsa_key = next(
                {
                    'kty': k['kty'],
                    'kid': k['kid'],
                    'use': k['use'],
                    'n': k['n'],
                    'e': k['e']
                }
                for k in jwks if k['kid'] == unverified_header['kid']
            )
            payload = jwt.decode(
                token,
                rsa_key,
                algorithms=['RS256'],
                audience=CLIENT_ID,
                issuer=f'https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{USER_POOL_ID}'
            )
            request.user = payload
        except Exception as e:
            return jsonify({'message': 'Token is invalid!', 'error': str(e)}), 401
        return f(*args, **kwargs)
    return decorated

# ------------------------------------------------------------------------------
# CEIPAL API integration
# ------------------------------------------------------------------------------
class CeipalAuth:
    def __init__(self):
        self.auth_url = "https://api.ceipal.com/v1/createAuthtoken/"
        self.email = os.getenv("CEIPAL_EMAIL")
        self.password = os.getenv("CEIPAL_PASSWORD")
        self.api_key = os.getenv("CEIPAL_API_KEY")
        self.token = None
        self.token_expiry = None

    def authenticate(self):
        payload = {
            "email": self.email,
            "password": self.password,
            "api_key": self.api_key,
            "json": "1"
        }
        try:
            r = requests.post(self.auth_url, json=payload)
            r.raise_for_status()
            data = r.json()
            if "access_token" in data:
                self.token = data["access_token"]
                self.token_expiry = datetime.now() + timedelta(hours=1)
                return True
        except Exception as e:
            logger.error(f"CEIPAL auth error: {e}")
        return False

    def get_token(self):
        if not self.token or datetime.now() >= self.token_expiry:
            if not self.authenticate():
                return None
        return self.token

class CeipalJobPostingsAPI:
    def __init__(self, auth):
        self.auth = auth
        self.base_url = "https://api.ceipal.com"
        self.job_postings_endpoint = "/getCustomJobPostingDetails/Z3RkUkt2OXZJVld2MjFpOVRSTXoxZz09/e6e04af381e7f42eeb7f942c8bf5ab6d"
        self.job_details_endpoint = "/v1/getJobPostingDetails/"

    def get_job_postings(self, paging_length=20):
        token = self.auth.get_token()
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(
            f"{self.base_url}{self.job_postings_endpoint}",
            headers=headers,
            params={"paging_length": paging_length}
        )
        return resp.json().get("results", [])

    def get_job_details(self, job_code):
        token = self.auth.get_token()
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(
            f"{self.base_url}{self.job_details_endpoint}",
            headers=headers,
            params={"job_id": job_code}
        )
        return resp.json()

# ------------------------------------------------------------------------------
# Semantic search setup
# ------------------------------------------------------------------------------
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
weight_multiplier = 1.0
FEEDBACK_THRESHOLD = 10

def nlrga_grade(score):
    return 'A' if score >= 85 else 'B' if score >= 70 else 'C' if score >= 50 else 'D'

def get_updated_multiplier():
    try:
        items = feedback_table.scan().get('Items', [])
        good = [i['Score'] for i in items if i.get('Feedback') == 'good']
        bad = [i['Score'] for i in items if i.get('Feedback') == 'bad']
        if good and bad:
            return min(max((np.mean(good) - np.mean(bad)) / 20 + 1, 0.5), 1.5)
    except:
        pass
    return 1.0

def retrain_embeddings():
    items = []
    scan = resume_table.scan()
    while True:
        items.extend(scan.get('Items', []))
        if 'LastEvaluatedKey' in scan:
            scan = resume_table.scan(ExclusiveStartKey=scan['LastEvaluatedKey'])
        else:
            break
    for item in items:
        text = item.get('ResumeText', '') + ' ' + ' '.join(item.get('Skills', []))
        if text:
            emb = model.encode(text).tolist()
            resume_table.update_item(
                Key={'email': item['email']},
                UpdateExpression='SET embedding = :e',
                ExpressionAttributeValues={':e': emb}
            )

@lru_cache(maxsize=50)
def get_user_embedding_cached(query):
    return model.encode(query)

def is_valid_item(item):
    required_fields = ['FullName', 'email', 'phone', 'Skills', 'Experience', 'SourceURL']
    return all(item.get(f) not in (None, '', []) for f in required_fields)

def clean_job_query(text):
    text = re.sub(r"(?i)job\s*code\s*:[^,\n]*", "", text)
    text = re.sub(r"(?i)location\s*:[^,\n]*", "", text)
    text = re.sub(r"(?i)job\s*type\s*:[^,\n]*", "", text)
    return ' '.join(w for w in text.strip().split() if w.lower() not in ["job", "code", "location", "type", "n/a", "jpc", "-", ":"])

def semantic_search(user_input, top_k=10):
    global weight_multiplier
    weight_multiplier = get_updated_multiplier()
    query = clean_job_query(user_input)
    user_emb = get_user_embedding_cached(query)

    docs, embs = [], []
    scan = resume_table.scan()
    while True:
        for it in scan.get('Items', []):
            if 'embedding' in it and is_valid_item(it):
                docs.append(it)
                embs.append(np.array(it['embedding']))
        if 'LastEvaluatedKey' in scan:
            scan = resume_table.scan(ExclusiveStartKey=scan['LastEvaluatedKey'])
        else:
            break

    if not docs:
        return [], "No resumes in database."

    matrix = np.vstack(embs)
    scores = np.dot(matrix, user_emb) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(user_emb))
    top = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for doc, sc in top:
        exp = int(doc.get('Experience') or 0)
        sc_int = max(1, min(int(sc * 100 * weight_multiplier), 100))
        results.append({
            'FullName': doc['FullName'],
            'email': doc['email'],
            'phone': doc['phone'],
            'Skills': doc['Skills'],
            'Experience': f"{exp} years",
            'SourceURL': doc['SourceURL'],
            'Score': sc_int,
            'Grade': nlrga_grade(sc_int)
        })
    return results, f"Top {len(results)} candidates found."

def keyword_search(user_input, top_k=10):
    q = clean_job_query(user_input).lower().split()
    results = []
    scan = resume_table.scan()
    while True:
        for it in scan.get('Items', []):
            text = (it.get('ResumeText', '') + ' ' + ' '.join(it.get('Skills', []))).lower()
            matches = sum(1 for w in q if w in text)
            if matches and is_valid_item(it):
                score = min(matches * 10, 100)
                results.append({
                    'FullName': it['FullName'],
                    'email': it['email'],
                    'phone': it['phone'],
                    'Skills': it['Skills'],
                    'Experience': f"{it['Experience']} years",
                    'SourceURL': it['SourceURL'],
                    'Score': score,
                    'Grade': nlrga_grade(score)
                })
        if 'LastEvaluatedKey' in scan:
            scan = resume_table.scan(ExclusiveStartKey=scan['LastEvaluatedKey'])
        else:
            break
    return sorted(results, key=lambda x: x['Score'], reverse=True)[:top_k], f"Found {len(results)} keyword matches."

def record_feedback(email, score, feedback):
    feedback_table.put_item(Item={"CandidateEmail": email, "Score": score, "Feedback": feedback})
    threading.Thread(target=retrain_embeddings).start()

# ------------------------------------------------------------------------------
# Routes: Authentication & User Management
# ------------------------------------------------------------------------------
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email'); password = data.get('password')
    fn = data.get('firstName'); ln = data.get('lastName')
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    full = f"{fn or ''} {ln or ''}".strip()
    attrs = [{'Name':'email','Value':email}]
    if fn: attrs.append({'Name':'given_name','Value':fn})
    if ln: attrs.append({'Name':'family_name','Value':ln})
    if full: attrs.append({'Name':'name','Value':full})
    try:
        resp = cognito_client.sign_up(
            ClientId=CLIENT_ID,
            SecretHash=get_secret_hash(email),
            Username=email,
            Password=password,
            UserAttributes=attrs
        )
        return jsonify({
            'message': 'User registered. Check email for code.',
            'userSub': resp['UserSub'],
            'codeDeliveryDetails': resp.get('CodeDeliveryDetails', {})
        }), 201
    except ClientError as e:
        code = e.response['Error']['Code']
        if code == 'UsernameExistsException':
            return jsonify({'error':'User already exists'}), 409
        return jsonify({'error': e.response['Error']['Message']}), 400

@app.route('/confirm', methods=['POST'])
def confirm_signup():
    data = request.get_json()
    email = data.get('email'); code = data.get('code')
    if not email or not code:
        return jsonify({'error':'Email and confirmation code are required'}), 400
    try:
        cognito_client.confirm_sign_up(
            ClientId=CLIENT_ID,
            SecretHash=get_secret_hash(email),
            Username=email,
            ConfirmationCode=code
        )
        return jsonify({'message':'Email confirmed successfully'}), 200
    except ClientError as e:
        ec = e.response['Error']['Code']
        if ec == 'ExpiredCodeException':
            return jsonify({'error':'Code expired'}), 400
        if ec == 'CodeMismatchException':
            return jsonify({'error':'Invalid code'}), 400
        if ec == 'UserNotFoundException':
            return jsonify({'error':'User not found'}), 404
        return jsonify({'error': e.response['Error']['Message']}), 400

@app.route('/resend-confirmation', methods=['POST'])
def resend_confirmation_code():
    data = request.get_json(); email = data.get('email')
    if not email:
        return jsonify({'error':'Email is required'}), 400
    try:
        cognito_client.resend_confirmation_code(
            ClientId=CLIENT_ID,
            SecretHash=get_secret_hash(email),
            Username=email
        )
        return jsonify({'message':'Confirmation code resent'}), 200
    except ClientError as e:
        ec = e.response['Error']['Code']
        if ec == 'UserNotFoundException':
            return jsonify({'error':'User not found'}), 404
        if ec == 'InvalidParameterException':
            return jsonify({'error':'Invalid email format'}), 400
        if ec == 'LimitExceededException':
            return jsonify({'error':'Too many attempts'}), 429
        return jsonify({'error': e.response['Error']['Message']}), 400

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json(); email = data.get('email'); password = data.get('password')
    if not email or not password:
        return jsonify({'error':'Email and password are required'}), 400
    try:
        resp = cognito_client.initiate_auth(
            ClientId=CLIENT_ID,
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={
                'USERNAME': email,
                'PASSWORD': password,
                'SECRET_HASH': get_secret_hash(email)
            }
        )
        ar = resp['AuthenticationResult']
        return jsonify({
            'access_token': ar['AccessToken'],
            'id_token': ar['IdToken'],
            'refresh_token': ar['RefreshToken']
        }), 200
    except ClientError as e:
        ec = e.response['Error']['Code']
        if ec == 'NotAuthorizedException':
            return jsonify({'error':'Invalid credentials'}), 401
        if ec == 'UserNotConfirmedException':
            return jsonify({'error':'User not confirmed'}), 401
        return jsonify({'error': e.response['Error']['Message']}), 400

@app.route('/secure-data', methods=['GET'])
@token_required
def secure_data():
    u = request.user
    email = u.get('email', u.get('sub'))
    return jsonify({
        'message': 'This is secure data',
        'user_info': {
            'sub': u.get('sub'),
            'email': email,
            'firstName': u.get('given_name', ''),
            'lastName': u.get('family_name', ''),
            'permissions': {
                "canViewAnalytics": True,
                "canManageUsers": True,
                "canManageRoles": True
            }
        }
    }), 200

@app.route('/user-groups', methods=['GET'])
@token_required
def get_user_groups():
    try:
        username = request.user['username']
        resp = cognito_client.admin_list_groups_for_user(
            Username=username,
            UserPoolId=USER_POOL_ID
        )
        return jsonify({'groups':[g['GroupName'] for g in resp['Groups']]}), 200
    except Exception as e:
        return jsonify({'error':str(e)}), 500

@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    data = request.get_json(); email = data.get('email')
    if not email:
        return jsonify({'error':'Email is required'}), 400
    cognito_client.forgot_password(
        ClientId=CLIENT_ID,
        SecretHash=get_secret_hash(email),
        Username=email
    )
    return jsonify({'message':'Reset code sent'}), 200

@app.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()
    email = data.get('email'); code = data.get('code'); new_pw = data.get('new_password')
    if not all([email, code, new_pw]):
        return jsonify({'error':'Email, code, and new password are required'}), 400
    cognito_client.confirm_forgot_password(
        ClientId=CLIENT_ID,
        SecretHash=get_secret_hash(email),
        Username=email,
        ConfirmationCode=code,
        Password=new_pw
    )
    return jsonify({'message':'Password reset successfully'}), 200

@app.route('/refresh-token', methods=['POST'])
def refresh_token():
    data = request.get_json(); rt = data.get('refresh_token')
    if not rt:
        return jsonify({'error':'Refresh token is required'}), 400
    resp = cognito_client.initiate_auth(
        ClientId=CLIENT_ID,
        AuthFlow='REFRESH_TOKEN_AUTH',
        AuthParameters={
            'REFRESH_TOKEN': rt,
            'SECRET_HASH': get_secret_hash('dummyuser@example.com')
        }
    )
    ar = resp['AuthenticationResult']
    return jsonify({'access_token': ar['AccessToken'], 'id_token':ar['IdToken']}), 200

@app.route('/logout', methods=['POST'])
@token_required
def logout():
    data = request.get_json(); rt = data.get('refresh_token')
    if not rt:
        return jsonify({'error':'Refresh token is required'}), 400
    cognito_client.revoke_token(
        ClientId=CLIENT_ID,
        ClientSecret=CLIENT_SECRET,
        Token=rt
    )
    return jsonify({'message':'Successfully logged out'}), 200

# ------------------------------------------------------------------------------
# Email config & test
# ------------------------------------------------------------------------------
@app.route('/test-email', methods=['POST'])
def test_email():
    data = request.get_json(); email = data.get('email')
    if not email:
        return jsonify({'error':'Email is required'}), 400
    up = cognito_client.describe_user_pool(UserPoolId=USER_POOL_ID)['UserPool']
    ec = up.get('EmailConfiguration', {})
    return jsonify({
        'message':'Email configuration details',
        'email_config':{
            'from': ec.get('From'),
            'source_arn': ec.get('SourceArn'),
            'reply_to': ec.get('ReplyToEmailAddress'),
            'configuration_set': ec.get('ConfigurationSet')
        },
        'user_pool_settings':{
            'auto_verified_attributes': up.get('AutoVerifiedAttributes', []),
            'verification_message_template': up.get('VerificationMessageTemplate',{})
        }
    }),200

@app.route('/check-email-config', methods=['GET'])
def check_email_config():
    up = cognito_client.describe_user_pool(UserPoolId=USER_POOL_ID)['UserPool']
    ec = up.get('EmailConfiguration',{})
    vt = up.get('VerificationMessageTemplate',{})
    return jsonify({
        'email_configuration':{
            'from': ec.get('From'),
            'source_arn': ec.get('SourceArn'),
            'reply_to': ec.get('ReplyToEmailAddress'),
            'configuration_set': ec.get('ConfigurationSet')
        },
        'verification_settings':{
            'default_email_option': vt.get('DefaultEmailOption'),
            'email_subject': vt.get('EmailSubject'),
            'email_message': vt.get('EmailMessage')
        },
        'auto_verified_attributes': up.get('AutoVerifiedAttributes',[]),
        'admin_create_user_config': up.get('AdminCreateUserConfig',{})
    }),200

# ------------------------------------------------------------------------------
# Resume upload
# ------------------------------------------------------------------------------
@app.route('/upload-resume', methods=['POST'])
@token_required
def upload_resume():
    user_id = request.user['sub']
    name = request.form.get('name'); email = request.form.get('email')
    phone = request.form.get('phone'); skills = request.form.get('skills')
    experience = request.form.get('experience')
    if 'file' not in request.files:
        return jsonify({'error':'No file provided'}), 400
    file = request.files['file']
    ext = os.path.splitext(file.filename)[1]
    unique_filename = f"resume-in-rest/{user_id}/{uuid.uuid4()}{ext}"
    try:
        s3_client.upload_fileobj(file, S3_BUCKET, unique_filename,
                                ExtraArgs={'ContentType': file.content_type})
    except Exception as e:
        return jsonify({'error':f'Failed S3 upload: {e}'}),500
    url = f"https://{S3_BUCKET}.s3.{COGNITO_REGION}.amazonaws.com/{unique_filename}"
    item = {
        'id': str(uuid.uuid4()),
        'userId': user_id,
        'filename': file.filename,
        'name': name,
        'email': email,
        'phone': phone,
        'skills': skills,
        'experience': experience,
        'resumeUrl': url,
        'uploadDate': datetime.utcnow().isoformat(),
        'status': 'active'
    }
    try:
        resume_table.put_item(Item=item, ConditionExpression='attribute_not_exists(id)')
    except Exception as e:
        return jsonify({'error':f'DynamoDB error: {e}'}),500
    return jsonify({'message':'Resume uploaded successfully','resumeId':item['id'],'resumeUrl':url}),201

# ------------------------------------------------------------------------------
# Stripe payment endpoints
# ------------------------------------------------------------------------------
@app.route('/create-payment-intent', methods=['POST'])
@token_required
def create_payment_intent():
    d = request.get_json()
    plan_id = d.get('metadata',{}).get('plan_id')
    billing = d.get('metadata',{}).get('billing_period')
    if not plan_id or not billing:
        return jsonify({'error':'Plan ID and billing period are required'}),400
    if plan_id not in STRIPE_PLANS or billing not in STRIPE_PLANS[plan_id]:
        return jsonify({'error':'Invalid plan or billing period'}),400
    pd = STRIPE_PLANS[plan_id][billing]
    if pd['amount'] < 50:
        return jsonify({'error':f"Amount must be ≥ $0.50, current ${pd['amount']/100:.2f}"}),400
    intent = stripe.PaymentIntent.create(
        amount=pd['amount'],
        currency='usd',
        metadata={
            'user_id': request.user['sub'],
            'plan_id': plan_id,
            'billing_period': billing,
            'price_id': pd['price_id']
        },
        automatic_payment_methods={'enabled': True, 'allow_redirects': 'always'}
    )
    return jsonify({
        'clientSecret': intent.client_secret,
        'paymentIntentId': intent.id,
        'status': intent.status
    }),200

@app.route('/payment-success', methods=['GET'])
@token_required
def payment_success():
    pid = request.args.get('payment_intent')
    if not pid:
        return jsonify({'error':'Payment intent ID is required'}),400
    pi = stripe.PaymentIntent.retrieve(pid)
    if pi.metadata.get('user_id') != request.user['sub']:
        return jsonify({'error':'Unauthorized access to payment'}),403
    handle_successful_payment(pi)
    return jsonify({
        'status':'success',
        'payment_intent':{
            'id':pi.id,'amount':pi.amount,'status':pi.status,'created':pi.created
        }
    }),200

@app.route('/payment-cancel', methods=['GET'])
@token_required
def payment_cancel():
    pid = request.args.get('payment_intent')
    if not pid:
        return jsonify({'error':'Payment intent ID is required'}),400
    pi = stripe.PaymentIntent.retrieve(pid)
    if pi.metadata.get('user_id') != request.user['sub']:
        return jsonify({'error':'Unauthorized access to payment'}),403
    if pi.status in ['requires_payment_method','requires_confirmation']:
        pi.cancel()
    return jsonify({'status':'cancelled','payment_intent':{'id':pi.id,'status':pi.status}}),200

def handle_successful_payment(pi):
    uid = pi.metadata.get('user_id')
    amt = pi.amount / 100
    resume_table.update_item(
        Key={'userId':uid},
        UpdateExpression='SET subscription_status=:s,payment_id=:p,last_payment_date=:d,last_payment_amount=:a',
        ExpressionAttributeValues={
            ':s':'active',':p':pi.id,':d':datetime.utcnow().isoformat(),':a':amt
        }
    )
    logger.info(f"Processed payment {pi.id} for user {uid}")

def handle_failed_payment(pi):
    uid = pi.metadata.get('user_id')
    err = pi.last_payment_error.get('message','Payment failed')
    resume_table.update_item(
        Key={'userId':uid},
        UpdateExpression='SET subscription_status=:s,payment_error=:e,last_payment_attempt=:d',
        ExpressionAttributeValues={
            ':s':'failed',':e':err,':d':datetime.utcnow().isoformat()
        }
    )
    logger.info(f"Payment {pi.id} failed for {uid}: {err}")

@app.route('/create-subscription', methods=['POST'])
@token_required
def create_subscription():
    d = request.get_json(); price_id = d.get('priceId')
    if not price_id:
        return jsonify({'error':'Price ID is required'}),400
    customer = stripe.Customer.create(
        email=request.user.get('email'),
        metadata={'user_id':request.user['sub']}
    )
    sub = stripe.Subscription.create(
        customer=customer.id,
        items=[{'price':price_id}],
        payment_behavior='default_incomplete',
        expand=['latest_invoice.payment_intent'],
        metadata={'user_id':request.user['sub']}
    )
    return jsonify({
        'subscriptionId': sub.id,
        'clientSecret': sub.latest_invoice.payment_intent.client_secret
    }),200

@app.route('/webhook', methods=['POST'])
def stripe_webhook():
    payload = request.get_data(as_text=True)
    sig = request.headers.get('Stripe-Signature')
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
        logger.info(f"Received webhook event: {event['type']}")
    except ValueError as e:
        return jsonify({'error':'Invalid payload'}),400
    except stripe.error.SignatureVerificationError as e:
        return jsonify({'error':'Invalid signature'}),400

    evt = event['data']['object']
    if event['type']=='payment_intent.succeeded':
        handle_successful_payment(evt)
    elif event['type']=='payment_intent.payment_failed':
        handle_failed_payment(evt)
    elif event['type']=='customer.subscription.created':
        handle_new_subscription(evt)
    elif event['type']=='customer.subscription.updated':
        handle_subscription_update(evt)
    elif event['type']=='customer.subscription.deleted':
        handle_subscription_cancellation(evt)
    else:
        logger.warning(f"Unhandled event {event['type']}")
    return jsonify({'status':'success'}),200

@app.route('/get-subscription', methods=['GET'])
@token_required
def get_subscription():
    customers = stripe.Customer.list(email=request.user.get('email'), limit=1)
    if not customers.data:
        return jsonify({'error':'No subscription found'}),404
    subs = stripe.Subscription.list(customer=customers.data[0].id, status='active', limit=1)
    if not subs.data:
        return jsonify({"error":"No active subscription found"}),404
    s = subs.data[0]
    return jsonify({
        'subscription':{
            'id':s.id,
            'status':s.status,
            'current_period_end':s.current_period_end,
            'cancel_at_period_end':s.cancel_at_period_end,
            'items':s.items.data
        }
    }),200

@app.route('/cancel-subscription', methods=['POST'])
@token_required
def cancel_subscription():
    d = request.get_json(); sid = d.get('subscriptionId')
    if not sid:
        return jsonify({'error':'Subscription ID is required'}),400
    sub = stripe.Subscription.modify(sid, cancel_at_period_end=True)
    return jsonify({
        'message':'Will cancel at period end',
        'subscription':{
            'id':sub.id,
            'cancel_at_period_end':sub.cancel_at_period_end,
            'current_period_end':sub.current_period_end
        }
    }),200

def handle_new_subscription(sub):
    uid = sub.metadata.get('user_id')
    resume_table.update_item(
        Key={'userId':uid},
        UpdateExpression='SET subscription_id=:i,subscription_status=:s,subscription_start=:st,subscription_end=:en',
        ExpressionAttributeValues={
            ':i':sub.id,
            ':s':sub.status,
            ':st':datetime.fromtimestamp(sub.current_period_start).isoformat(),
            ':en':datetime.fromtimestamp(sub.current_period_end).isoformat()
        }
    )
    logger.info(f"New subscription {sub.id} for user {uid}")

def handle_subscription_update(sub):
    uid = sub.metadata.get('user_id')
    resume_table.update_item(
        Key={'userId':uid},
        UpdateExpression='SET subscription_status=:s,subscription_end=:en,cancel_at_period_end=:ca',
        ExpressionAttributeValues={
            ':s':sub.status,
            ':en':datetime.fromtimestamp(sub.current_period_end).isoformat(),
            ':ca':sub.cancel_at_period_end
        }
    )
    logger.info(f"Subscription {sub.id} updated for {uid}")

def handle_subscription_cancellation(sub):
    uid = sub.metadata.get('user_id')
    resume_table.update_item(
        Key={'userId':uid},
        UpdateExpression='SET subscription_status=:s,subscription_end=:en',
        ExpressionAttributeValues={
            ':s':'cancelled',
            ':en':datetime.fromtimestamp(sub.current_period_end).isoformat()
        }
    )
    logger.info(f"Subscription {sub.id} cancelled for {uid}")

@app.route('/create-checkout-session', methods=['POST'])
@token_required
def create_checkout_session():
    d = request.get_json()
    plan_id = d.get('planId'); billing = d.get('billingPeriod')
    success_url = d.get('successUrl'); cancel_url = d.get('cancelUrl')
    if not plan_id or not billing:
        return jsonify({'error':'Plan ID and billing period are required'}),400
    if plan_id not in STRIPE_PLANS or billing not in STRIPE_PLANS[plan_id]:
        return jsonify({'error':'Invalid plan or billing period'}),400
    pd = STRIPE_PLANS[plan_id][billing]
    session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[{'price':pd['price_id'],'quantity':1}],
        mode='subscription',
        success_url=success_url,
        cancel_url=cancel_url,
        allow_promotion_codes=True,
        metadata={'user_id':request.user['sub'],'plan_id':plan_id,'billing_period':billing}
    )
    return jsonify({'url':session.url}),200

@app.route('/activate-free-plan', methods=['POST'])
@token_required
def activate_free_plan():
    uid = request.user['sub']
    resume_table.update_item(
        Key={'userId':uid},
        UpdateExpression='SET subscription_status=:s,plan_id=:p,last_payment_date=:d',
        ExpressionAttributeValues={
            ':s':'active',':p':'test',':d':datetime.utcnow().isoformat()
        }
    )
    return jsonify({'message':'Free plan activated'}),200

# ------------------------------------------------------------------------------
# Routes: Frontend & CEIPAL endpoints
# ------------------------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    q = data.get('query',''); algo = data.get('algorithm','semantic')
    if not q:
        return jsonify({'error':'Empty query'}),400
    if algo=='keyword':
        res, summ = keyword_search(q)
    else:
        res, summ = semantic_search(q)
        if not res:
            res, summ = keyword_search(q)
    return jsonify({'results':res,'summary':summ})

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    d = request.json
    record_feedback(d.get('email'), d.get('score'), d.get('feedback'))
    return jsonify({'status':'success'})

@app.route('/api/v1/ceipal/jobs', methods=['GET'])
def get_ceipal_jobs():
    try:
        jobs = CeipalJobPostingsAPI(CeipalAuth()).get_job_postings()
        return jsonify({"jobs":jobs})
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route('/api/v1/ceipal/getJobDetails', methods=['GET'])
def get_job_details():
    code = request.args.get('job_code')
    if not code:
        return jsonify({"error":"Missing job_code parameter"}),400
    try:
        job = CeipalJobPostingsAPI(CeipalAuth()).get_job_details(code)
        return jsonify({"job_details":job})
    except Exception as e:
        return jsonify({"error":str(e)}),500

# ------------------------------------------------------------------------------
# Run the app
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    if not all([os.getenv(v) for v in ["CEIPAL_EMAIL","CEIPAL_PASSWORD","CEIPAL_API_KEY"]]):
        logger.error("Missing CEIPAL credentials.")
    else:
        app.run(host='0.0.0.0', port=8000, debug=True)
