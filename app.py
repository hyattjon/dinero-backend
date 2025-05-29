from flask import Flask, request, jsonify
from flask_cors import CORS
from plaid2 import Client
from dotenv import load_dotenv
from flask_talisman import Talisman
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import pandas as pd
import requests
import datetime
import secrets
import logging
from logging.handlers import RotatingFileHandler

load_dotenv()

app = Flask(__name__)
# Enable HTTPS
Talisman(app)

# Secure configurations
app.config.update(
    SECRET_KEY=os.environ.get('FLASK_SECRET_KEY', secrets.token_urlsafe(32)),
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

# Update CORS to be more restrictive
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",  # Local development
            "https://cardmatcher-frontend-1dd33be51a4a.herokuapp.com"  # Your frontend URL
        ],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

PLAID_CLIENT_ID = os.environ.get("PLAID_CLIENT_ID")
PLAID_SECRET = os.environ.get("PLAID_SECRET")
PLAID_ENV = os.environ.get("PLAID_ENV", "sandbox")

client = Client(client_id=PLAID_CLIENT_ID, secret=PLAID_SECRET, environment=PLAID_ENV)

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

def fetch_credit_cards():
    url = "https://rewardscc.com/api/creditcards"
    resp = requests.get(url)
    if resp.ok:
        return resp.json()
    return []

def fetch_plaid_transactions(public_token):
    exchange_response = client.Item.public_token.exchange(public_token)
    access_token = exchange_response["access_token"]
    # Fetch transactions for the last year
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    response = client.Transactions.get(access_token, start_date, end_date)
    return response["transactions"]

def estimate_cashback(transactions, card):
    """
    Estimate cashback for a card given a list of transactions.
    This is a simple version: assumes a flat cashback rate.
    You can enhance this to use category-based rates if available in card data.
    """
    # Try to get the cashback rate from the card data, default to 1%
    cashback_rate = float(card.get('cashback', 0.01))
    total_cashback = 0.0
    for t in transactions:
        amt = abs(float(t.get("amount", 0)))
        total_cashback += amt * cashback_rate
    return round(total_cashback, 2)

def recommend_cards(transactions, credit_cards):
    # Calculate estimated cashback for each card
    for card in credit_cards:
        card['estimated_cashback'] = estimate_cashback(transactions, card)
    # Sort by estimated cashback, descending
    sorted_cards = sorted(credit_cards, key=lambda x: x['estimated_cashback'], reverse=True)
    return sorted_cards[:3]  # Top 3 cards

# Add this function after your other helper functions
def calculate_cashback(transactions):
    """Calculate total cashback for transactions using a default 1% rate"""
    total = 0
    for t in transactions:
        amount = abs(float(t.get("amount", 0)))
        total += amount * 0.01  # 1% cashback
    return round(total, 2)

# Setup error logging
if not app.debug:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Application startup')

# Add error handling to routes
@app.route("/recommend_cards", methods=["POST"])
@limiter.limit("5 per minute")
def recommend_cards_endpoint():
    try:
        public_token = request.json.get("public_token")
        if not public_token:
            return jsonify({"error": "Missing public_token"}), 400
            
        transactions = fetch_plaid_transactions(public_token)
        credit_cards = fetch_credit_cards()
        recommendations = recommend_cards(transactions, credit_cards)
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        app.logger.error(f'Error processing recommendation: {str(e)}')
        return jsonify({"error": "Internal server error"}), 500

@app.route("/get_access_token", methods=["POST"])
def get_access_token():
    public_token = request.json["public_token"]
    exchange_response = client.Item.public_token.exchange(public_token)
    access_token = exchange_response["access_token"]
    return jsonify({"access_token": access_token})

@app.route("/get_transactions", methods=["POST"])
def get_transactions():
    public_token = request.json.get("public_token")
    exchange_response = client.Item.public_token.exchange(public_token)
    access_token = exchange_response["access_token"]
    import datetime
    start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    response = client.Transactions.get(access_token, start_date, end_date)
    transactions = response["transactions"]
    cashback = calculate_cashback(transactions)
    return jsonify({"transactions": transactions, "cashback": cashback})

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    file = request.files["file"]
    df = pd.read_csv(file)
    transactions = df.to_dict(orient="records")
    cashback = calculate_cashback(transactions)
    return jsonify({"transactions": transactions, "cashback": cashback})

@app.route("/create_link_token", methods=["POST"])
@limiter.limit("5 per minute")
def create_link_token():
    user_id = request.json.get("user_id", "user-demo")  # You can use a real user ID if you have auth
    response = client.LinkToken.create({
        "user": {"client_user_id": user_id},
        "client_name": "Credit Card Recommender",
        "products": ["transactions"],
        "country_codes": ["US"],
        "language": "en",
        "redirect_uri": None  # Only needed for OAuth
    })
    return jsonify({"link_token": response["link_token"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))