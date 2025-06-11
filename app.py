from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask import make_response
import redis
import plaid
from plaid.api import plaid_api
from dotenv import load_dotenv
from flask_talisman import Talisman
from flask_wtf.csrf import CSRFProtect
import os
import pandas as pd
import requests
import datetime
import secrets
import logging
from logging.handlers import RotatingFileHandler
import json
import httpx
import asyncio
from typing import List, Dict, Any
import random
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from functools import wraps
from supabase import create_client, Client
import uuid
import stripe
from flask_jwt_extended import JWTManager, create_access_token, create_refresh_token, get_jwt_identity, jwt_required
import structlog
import traceback
import sys
from marshmallow import Schema, fields, validate, ValidationError
from html import escape

# This works now again
# Environment variables and configuration
load_dotenv()

# Define configuration object for Plaid and other services
CONFIG = {
    'PLAID_CLIENT_ID': os.environ.get('PLAID_CLIENT_ID'),
    'PLAID_SECRET': os.environ.get('PLAID_SECRET'),
    'PLAID_ENV': os.environ.get('PLAID_ENV', 'sandbox'),
    'PLAID_PRODUCTS': ['transactions'],
    'PLAID_COUNTRY_CODES': ['US'],
    'PLAID_WEBHOOK': os.environ.get('PLAID_WEBHOOK', ''),  # Optional webhook URL
    'PLAID_REDIRECT_URI': os.environ.get('PLAID_REDIRECT_URI', ''),
    'REWARDS_CC_API_KEY': os.environ.get('REWARDS_CC_API_KEY'),
    'REWARDS_CC_API_HOST': os.environ.get('REWARDS_CC_API_HOST'),
    'REWARDS_CC_BASE_URL': os.environ.get('REWARDS_CC_BASE_URL')
}

# RapidAPI configuration
REWARDS_CC_API_KEY = os.environ.get('REWARDS_CC_API_KEY')
REWARDS_CC_API_HOST = os.environ.get('REWARDS_CC_API_HOST')
REWARDS_CC_BASE_URL = os.environ.get('REWARDS_CC_BASE_URL')

# Supabase configuration
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

# JWT configuration
JWT_SECRET = os.environ.get('JWT_SECRET', secrets.token_urlsafe(32))
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '')

# Add with other configuration


# Stripe configuration
STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', '')
STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY', '')
STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET', '')
STRIPE_PRICE_BASIC = os.environ.get('STRIPE_PRICE_BASIC', '')
STRIPE_PRICE_PRO = os.environ.get('STRIPE_PRICE_PRO', '')

# Define subscription plans
PAYMENT_PLANS = {
    'free': {
        'name': 'Free',
        'price': 0,
        'features': [
            'Basic card recommendations',
            'Connect one bank account',
            'View top 3 card matches'
        ]
    },
    'basic': {
        'name': 'Basic',
        'price': 5,
        'stripe_price_id': STRIPE_PRICE_BASIC if STRIPE_PRICE_BASIC else None,
        'features': [
            'All Free features',
            'Connect multiple bank accounts',
            'View all card recommendations',
            'Detailed rewards analysis'
        ]
    },
    'pro': {
        'name': 'Pro',
        'price': 20,
        'stripe_price_id': STRIPE_PRICE_PRO if STRIPE_PRICE_PRO else None,
        'features': [
            'All Basic features',
            'Custom card scoring',
            'Advanced category analysis',
            'Annual spending projections',
            'Priority support'
        ]
    }
}


# Add debug logging
print("=== Environment Variables Debug ===")
print("Current environment variables:")
for key in ['REWARDS_CC_API_KEY', 'REWARDS_CC_API_HOST', 'REWARDS_CC_BASE_URL']:
    print(f"{key}: {os.getenv(key)}")
print("===============================")

# Check required variables
required_vars = [
    'REWARDS_CC_API_KEY',
    'REWARDS_CC_API_HOST',
    'REWARDS_CC_BASE_URL'
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    print("ERROR: Missing variables:", missing_vars)
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Simple in-memory user storage (replace with database in production)
users = {}

# Validate required environment variables
required_env_vars = {
    'PLAID_CLIENT_ID': CONFIG['PLAID_CLIENT_ID'],
    'PLAID_SECRET': CONFIG['PLAID_SECRET'],
    'REWARDS_CC_API_KEY': REWARDS_CC_API_KEY,
    'REWARDS_CC_API_HOST': REWARDS_CC_API_HOST,
    'REWARDS_CC_BASE_URL': REWARDS_CC_BASE_URL,
    'SUPABASE_URL': SUPABASE_URL,
    'SUPABASE_KEY': SUPABASE_KEY
}

missing_vars = [var for var, value in required_env_vars.items() if not value]
if missing_vars:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize clients
# Configure Plaid client
configuration = plaid.Configuration(
    host=plaid.Environment.Sandbox,
    api_key={
        'clientId': CONFIG['PLAID_CLIENT_ID'],
        'secret': CONFIG['PLAID_SECRET'],
        'plaidVersion': '2020-09-14'  # Add API version
    }
)

api_client = plaid.ApiClient(configuration)
client = plaid_api.PlaidApi(api_client)

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Stripe
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
else:
    print("Stripe secret key not found. Stripe functionality will be disabled.")

# Initialize Flask app and extensions
app = Flask(__name__)

# Set up CORS properly with specific origins
app.config['CORS_HEADERS'] = 'Content-Type,Authorization'

CORS(app, 
     resources={r"/*": {
         "origins": ["http://localhost:3000", 
                     "http://127.0.0.1:3000", 
                     "http://localhost:5001",
                     "http://127.0.0.1:5001",
                     "https://cardmatcher.net",
                     "https://www.cardmatcher.net"],
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"]
     }},
     supports_credentials=True)

# Disable SSL requirement for local development
app.config['TALISMAN_ENABLED'] = False

# Initialize Stripe client if key exists
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
else:
    app.logger.warning("Stripe secret key not found. Stripe functionality will be disabled.")

# Secure configurations
app.config.update(
    SECRET_KEY=os.environ.get('FLASK_SECRET_KEY', secrets.token_urlsafe(32)),
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

# Configure Redis and rate limiting
redis_url = os.environ.get('UPSTASH_REDIS_URL', 'memory://')
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=redis_url,
    storage_options={},
    default_limits=["200 per day", "50 per hour"]
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Custom JSON encoder for Flask
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        if isinstance(obj, plaid.model.transaction.Transaction):
            return self.serialize_plaid_transaction(obj)
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super().default(obj)

    def serialize_plaid_transaction(self, transaction):
        tx_dict = transaction.to_dict()
        # Convert date objects to ISO format strings
        for key, value in tx_dict.items():
            if isinstance(value, (datetime.date, datetime.datetime)):
                tx_dict[key] = value.isoformat()
        return tx_dict

# Helper functions
def fetch_credit_cards():
    url = "https://rewardscc.com/api/creditcards"
    resp = requests.get(url)
    if resp.ok:
        return resp.json()
    return []

def fetch_plaid_transactions(public_token):
    try:
        # Exchange public token for access token
        exchange_response = client.item_public_token_exchange({
            'public_token': public_token
        })
        access_token = exchange_response['access_token']
        
        # Get transactions
        start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        transactions_response = client.transactions_get(
            transactions_get_request={
                'access_token': access_token,
                'start_date': start_date,
                'end_date': end_date,
                'options': {
                    'count': 100,
                    'offset': 0
                }
            }
        )
        
        return transactions_response['transactions']
    except plaid.ApiException as e:
        app.logger.error(f"Plaid API error: {str(e)}")
        raise

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
    """Compare and recommend credit cards based on spending patterns"""
    card_analysis = []
    
    # Calculate total spending for reference
    total_spending = sum(abs(float(t.get('amount', 0))) for t in transactions)
    
    for card in credit_cards:
        # Calculate rewards for this card
        rewards_analysis = calculate_card_rewards(transactions, card)
        
        card_analysis.append({
            'card_name': card.get('name', 'Unknown Card'),
            'card_id': card.get('id'),
            'issuer': card.get('issuer'),
            'annual_fee': card.get('annual_fee', 0),
            'rewards_earned': rewards_analysis['total_rewards'],
            'effective_rewards_rate': rewards_analysis['effective_rate'],
            'category_breakdown': rewards_analysis['category_rewards'],
            'net_rewards': rewards_analysis['total_rewards'] - float(card.get('annual_fee', 0)),
            'signup_bonus': card.get('signup_bonus'),
            'signup_bonus_requirement': card.get('signup_bonus_spend_requirement'),
            'card_details': card
        })
    
    # Sort by net rewards (rewards minus annual fee)
    sorted_cards = sorted(card_analysis, key=lambda x: x['net_rewards'], reverse=True)
    
    # Add comparison metrics
    best_card = sorted_cards[0]
    for card in sorted_cards:
        card['potential_savings'] = round(card['net_rewards'] - best_card['net_rewards'], 2)
    
    return sorted_cards[:3]  # Top 3 cards

def calculate_cashback(transactions):
    """Calculate total cashback for transactions using a default 1% rate"""
    total = 0
    for t in transactions:
        amount = abs(float(t.get("amount", 0)))
        total += amount * 0.01  # 1% cashback
    return round(total, 2)

def calculate_card_rewards(transactions, card):
    """Calculate rewards using detailed Plaid categories"""
    total_rewards = 0
    category_rewards = {}
    running_totals = {}  # Track spending for limits
    
    # Get base earn rate
    base_rate = float(card.get('baseSpendAmount', 1.0))
    base_valuation = float(card.get('baseSpendEarnValuation', 1.0))
    
    # Get Plaid reward structure
    plaid_rewards = card.get('plaid_rewards', {})
    detailed_rewards = {
        reward['plaidDetailed']: reward 
        for reward in plaid_rewards.get('plaidDetailed', [])
    }
    
    for transaction in transactions:
        amount = abs(float(transaction.get('amount', 0)))
        
        # Get transaction category
        pfc = transaction.get('personal_finance_category', {})
        detailed_category = pfc.get('detailed') if pfc else None
        
        # Find applicable reward rate
        reward_rate = base_rate
        category_name = 'Base Spend'
        
        if detailed_category in detailed_rewards:
            reward_info = detailed_rewards[detailed_category]
            
            # Check spend limits if applicable
            if reward_info.get('isSpendLimit'):
                limit = float(reward_info.get('spendLimit', 0))
                period = reward_info.get('spendLimitResetPeriod', 'Year')
                
                # Track running total
                running_totals.setdefault(detailed_category, 0)
                if running_totals[detailed_category] < limit:
                    reward_rate = float(reward_info.get('earnMultiplier', base_rate))
                    category_name = detailed_category
                    
                running_totals[detailed_category] += amount
            else:
                reward_rate = float(reward_info.get('earnMultiplier', base_rate))
                category_name = detailed_category
        
        # Calculate rewards
        reward = amount * reward_rate * base_valuation
        total_rewards += reward
        
        # Track by category
        if category_name not in category_rewards:
            category_rewards[category_name] = 0
        category_rewards[category_name] += reward

    return {
        'total_rewards': round(total_rewards, 2),
        'category_rewards': {k: round(v, 2) for k, v in category_rewards.items()},
        'effective_rate': round(total_rewards / sum(abs(float(t.get('amount', 0))) for t in transactions) * 100, 2),
        'spending_limits': {k: {'spent': v, 'limit': detailed_rewards[k].get('spendLimit')} 
                          for k, v in running_totals.items() if k in detailed_rewards}
    }

def analyze_transactions(transactions):
    """Analyze transactions by category and return spending summary"""
    category_totals = {}
    total_spent = 0
    
    try:
        for transaction in transactions:
            # Use personal_finance_category if available, fall back to category, or use 'Other'
            categories = []
            if transaction.get('personal_finance_category'):
                pfc = transaction['personal_finance_category']
                categories = [pfc['primary'], pfc.get('detailed', 'General')]
            elif transaction.get('category'):
                categories = transaction['category']
            else:
                categories = ['Other', 'Uncategorized']

            main_category = categories[0]
            
            # Get transaction amount (absolute value since expenses are negative)
            amount = abs(float(transaction.get('amount', 0)))
            total_spent += amount
            
            # Initialize category if not exists
            if main_category not in category_totals:
                category_totals[main_category] = {
                    'total': 0,
                    'subcategories': {}
                }
            
            # Update category total
            category_totals[main_category]['total'] += amount
            
            # Handle subcategory
            sub_category = categories[1] if len(categories) > 1 else 'General'
            if sub_category not in category_totals[main_category]['subcategories']:
                category_totals[main_category]['subcategories'][sub_category] = 0
            category_totals[main_category]['subcategories'][sub_category] += amount

        app.logger.info(f"Analysis completed: {len(category_totals)} categories found")
        
        return {
            'category_totals': category_totals,
            'total_spent': total_spent,
            'transaction_count': len(transactions),
            'categories': list(category_totals.keys())
        }
    except Exception as e:
        app.logger.error(f"Error in analyze_transactions: {str(e)}")
        return {
            'category_totals': {'Other': {'total': 0, 'subcategories': {'Uncategorized': 0}}},
            'total_spent': sum(abs(float(t.get('amount', 0))) for t in transactions),
            'transaction_count': len(transactions),
            'categories': ['Other']
        }

def get_sample_transactions():
    """Generate realistic sample transactions"""
    sample_transactions = []
    # Common merchant categories
    merchants = [
        {"name": "Whole Foods", "category": "GROCERY_STORES"},
        {"name": "Shell Gas", "category": "GAS_STATIONS"},
        {"name": "Amazon", "category": "ONLINE_SHOPPING"},
        {"name": "Starbucks", "category": "RESTAURANTS"},
        {"name": "Netflix", "category": "ENTERTAINMENT"},
        {"name": "Target", "category": "RETAIL_STORES"}
    ]
    
    # Generate 30 days of transactions
    for i in range(30):
        date = (datetime.datetime.now() - datetime.timedelta(days=i)).strftime('%Y-%m-%d')
        # Generate 2-4 transactions per day
        for _ in range(random.randint(2, 4)):
            merchant = random.choice(merchants)
            amount = round(random.uniform(10, 200), 2)
            
            transaction = {
                "transaction_id": f"sample-{date}-{_}",
                "amount": -amount,  # Negative for expenses
                "date": date,
                "merchant_name": merchant["name"],
                "personal_finance_category": {
                    "primary": merchant["category"],
                    "detailed": merchant["category"]
                }
            }
            sample_transactions.append(transaction)
    
    return sample_transactions

async def fetch_card_details(card_key: str, client: httpx.AsyncClient) -> Dict[Any, Any]:
    """Fetch card details for a specific card key"""
    if not all([REWARDS_CC_BASE_URL, REWARDS_CC_API_KEY, REWARDS_CC_API_HOST]):
        app.logger.error("Missing required RapidAPI configuration")
        return {card_key: None}

    url = f"{REWARDS_CC_BASE_URL}/creditcard-plaid-bycard/{card_key}"
    headers = {
        'X-RapidAPI-Key': REWARDS_CC_API_KEY,
        'X-RapidAPI-Host': REWARDS_CC_API_HOST
    }
    
    try:
        app.logger.info(f"Fetching card details for {card_key}")
        response = await client.get(
            url, 
            headers=headers, 
            timeout=10.0,
            follow_redirects=True
        )
        
        app.logger.info(f"Response status for {card_key}: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            app.logger.info(f"Successfully fetched data for {card_key}")
            return {card_key: data}
        else:
            app.logger.error(f"Error fetching {card_key}: {response.status_code}")
            app.logger.error(f"Response content: {response.text}")
            return {card_key: None}
    except Exception as e:
        app.logger.error(f"Exception fetching {card_key}: {str(e)}")
        return {card_key: None}

def fetch_card_details_sync(card_key: str) -> Dict[Any, Any]:
    """Fetch card details for a specific card key (synchronous version)"""
    try:
        # Check if API keys are available
        if not all([REWARDS_CC_API_KEY, REWARDS_CC_API_HOST]):
            app.logger.warning(f"Missing API credentials, using fallback data for {card_key}")
            return {"cardName": f"Sample Card ({card_key})", "cardIssuer": "Sample Bank", "annualFee": "0"}
            
        # Fix the URL format to use the proper RapidAPI structure
        url = f"https://{REWARDS_CC_API_HOST}/creditcard-plaid-bycard/{card_key}"
        
        headers = {
            'x-rapidapi-key': REWARDS_CC_API_KEY,
            'x-rapidapi-host': REWARDS_CC_API_HOST
        }
        
        app.logger.info(f"Fetching card details for {card_key} from {url}")
        response = requests.get(url, headers=headers, timeout=10)
        
        app.logger.info(f"Response status for {card_key}: {response.status_code}")
        
        if response.status_code == 200:
            app.logger.info(f"Successfully fetched data for {card_key}")
            return response.json()
        else:
            app.logger.error(f"Error fetching {card_key}: {response.status_code}")
            app.logger.error(f"Response content: {response.text}")
            # Return fallback data on error
            return {"cardName": f"Sample Card ({card_key})", "cardIssuer": "Sample Bank", "annualFee": "0"}
    except Exception as e:
        app.logger.error(f"Exception fetching {card_key}: {str(e)}")
        # Return fallback data on exception
        return {"cardName": f"Sample Card ({card_key})", "cardIssuer": "Sample Bank", "annualFee": "0"}

def calculate_card_potential_rewards(transactions, card_data):
    """Calculate potential rewards for a card based on actual transactions"""
    total_rewards = 0
    category_rewards = {}
    running_totals = {}
    
    # Handle case where card_data is a list (RapidAPI response format)
    if isinstance(card_data, list) and len(card_data) > 0:
        card_data = card_data[0]  # Take first item from list
    
    # Get base earn rate and multiplier
    base_rate = float(card_data.get('baseSpendAmount', 1.0))
    base_valuation = float(card_data.get('baseSpendEarnValuation', 1.0))
    
    # Get category-specific reward rates
    category_rates = {}
    plaid_detailed = card_data.get('plaidDetailed', [])
    if isinstance(plaid_detailed, list):
        for reward in plaid_detailed:
            if isinstance(reward, dict):
                category_rates[reward.get('plaidDetailed')] = {
                    'rate': float(reward.get('earnMultiplier', base_rate)),
                    'limit': float(reward.get('spendLimit', 0)) if reward.get('isSpendLimit') else None,
                    'period': reward.get('spendLimitResetPeriod')
                }
    
    for transaction in transactions:
        amount = abs(float(transaction.get('amount', 0)))
        category = transaction.get('personal_finance_category', {}).get('detailed')
        
        # Find best applicable reward rate
        best_rate = base_rate
        category_name = 'Base Spend'
        
        if category in category_rates:
            rate_info = category_rates[category]
            if rate_info['limit'] is None or running_totals.get(category, 0) < rate_info['limit']:
                best_rate = rate_info['rate']
                category_name = category
            
            if rate_info['limit']:
                running_totals[category] = running_totals.get(category, 0) + amount
        
        # Calculate reward
        reward = amount * best_rate * base_valuation
        total_rewards += reward
        
        # Track by category
        if category_name not in category_rewards:
            category_rewards[category_name] = 0
        category_rewards[category_name] += reward
    
    total_spend = sum(abs(float(t.get('amount', 0))) for t in transactions)
    
    return {
        'total_rewards': round(total_rewards, 2),
        'category_rewards': {k: round(v, 2) for k, v in category_rewards.items()},
        'effective_rate': round(total_rewards / total_spend * 100, 2) if total_spend > 0 else 0,
        'spending_limits': {k: {'spent': v, 'limit': category_rates[k]['limit']} 
                          for k, v in running_totals.items() if k in category_rates}
    }

# JWT Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        
        app.logger.info(f"Auth header: {auth_header}")
        
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        if not token:
            app.logger.error("Token is missing")
            return jsonify({"error": "Token is missing"}), 401
        
        try:
            # Verify token
            app.logger.info(f"Verifying token: {token[:10]}...")
            
            # First try our own JWT verification
            try:
                decoded_token = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
                app.logger.info(f"Decoded token: {decoded_token}")
                
                # Pass the user information to the decorated function
                return f({'id': decoded_token.get('id'), 'email': decoded_token.get('email')}, *args, **kwargs)
            except jwt.InvalidTokenError as e:
                # If our verification fails and the token might be a Supabase token, try to get the user ID from Supabase
                app.logger.warning(f"Our JWT verification failed, trying Supabase session: {str(e)}")
                
                # Extract user ID from Supabase token (this is just a backup)
                try:
                    # Use the token to get user info from Supabase
                    user = supabase.auth.get_user(token)
                    user_id = user.user.id
                    email = user.user.email
                    
                    # Generate our own token for future use
                    new_token = jwt.encode({
                        'id': user_id,
                        'email': email,
                        'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7)
                    }, JWT_SECRET, algorithm="HS256")
                    
                    # Return the function result with a header asking client to update token
                    response = f({'id': user_id, 'email': email}, *args, **kwargs)
                    if isinstance(response, tuple):
                        json_response, status_code = response
                        # Add token to response
                        json_response['new_token'] = new_token
                        return jsonify(json_response), status_code
                    else:
                        # Add token to response
                        response_data = response.get_json()
                        response_data['new_token'] = new_token
                        return jsonify(response_data)
                except Exception as supabase_error:
                    app.logger.error(f"Supabase token verification failed: {str(supabase_error)}")
                    return jsonify({"error": "Invalid token"}), 401
                
        except Exception as e:
            app.logger.error(f"Token verification error: {str(e)}")
            return jsonify({"error": "Authentication failed"}), 401
    
    return decorated

# Define ALL routes together
# Options handler for CORS
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    response = app.make_default_options_response()
    return response

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        origin = request.headers.get('Origin')
        
        allowed_origins = [
            "http://localhost:3000", 
            "http://127.0.0.1:3000",
            "http://localhost:5001", 
            "http://127.0.0.1:5001",
            "https://cardmatcher.net", 
            "https://www.cardmatcher.net"
        ]
        
        if origin in allowed_origins:
            response.headers.add('Access-Control-Allow-Origin', origin)
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS,PUT,DELETE')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
            response.headers.add('Cross-Origin-Opener-Policy', 'same-origin-allow-popups')
        
        return response, 200

@app.route("/auth/register", methods=["POST"])
@limiter.exempt
def register():
    try:
        data = request.get_json()
        app.logger.info(f"Registration attempt: {data.get('email') if data else 'No data'}")
        
        # Validate input with schema
        try:
            schema = RegistrationSchema()
            validated_data = schema.load(data)
        except ValidationError as err:
            return jsonify({"success": False, "error": "Validation error", "details": err.messages}), 400
        
        # Sanitize inputs
        name = sanitize_input(validated_data.get('name'))
        email = validated_data.get('email')  # Email already validated by schema
        password = validated_data.get('password')
        
        # Check if user exists
        user_query = supabase.table('users').select('*').eq('email', email).execute()
        app.logger.info(f"User query result: {len(user_query.data)} records found")

        if len(user_query.data) > 0:
            app.logger.info(f"Email {email} already registered")
            return jsonify({"success": False, "error": "Email already registered"}), 400
        
        # Register with Supabase Auth
        signup_data = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        
        # Store user metadata
        user_id = signup_data.user.id if signup_data and signup_data.user else None
        
        # Check if we got a valid user ID
        if not user_id:
            return jsonify({"success": False, "error": "Failed to create user account"}), 500
            
        # Auto-confirm the email (use admin API)
        try:
            # This requires service role key, not anon key
            admin_supabase = create_client(
                SUPABASE_URL, 
                os.environ.get('SUPABASE_SERVICE_ROLE')  # Use your env variable name
            )
            
            # Update the user's confirmation status directly 
            admin_supabase.auth.admin.update_user_by_id(
                user_id,
                {"email_confirm": True}  # This confirms the email
            )
            
            app.logger.info(f"Auto-confirmed email for: {email}")
        except Exception as e:
            app.logger.error(f"Could not auto-confirm email: {str(e)}")
            # Continue anyway - user will need to confirm manually
            
        # Insert user data into your users table
        supabase.table('users').insert({
            "id": user_id,
            "email": email,
            "name": name,
            "created_at": datetime.datetime.now().isoformat()
        }).execute()
        
        app.logger.info(f"User registered: {email}")
        
        return jsonify({
            "success": True,
            "message": "Account created. Please check your email to confirm registration."
        })
        
    except Exception as e:
        app.logger.error(f"Registration error: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": "Registration failed"}), 500

@app.route("/auth/login", methods=["POST"])
@limiter.limit("5 per minute")  # Specific rate limit for login attempts
def login():
    try:
        # Get login credentials
        data = request.get_json()
        
        # Validate input
        try:
            schema = LoginSchema()
            validated_data = schema.load(data)
        except ValidationError as err:
            return jsonify({"success": False, "error": "Validation error", "details": err.messages}), 400
        
        email = validated_data.get('email')
        password = validated_data.get('password')
        
        # Authenticate with Supabase
        login_response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        user_id = login_response.user.id
        
        # Update last login time
        supabase.table('users').update({
            "last_login": datetime.datetime.now().isoformat()
        }).eq('id', user_id).execute()
        
        # Get user name
        user_query = supabase.table('users').select('name').eq('id', user_id).execute()
        name = user_query.data[0]['name'] if user_query.data else email.split('@')[0]
        
        # Create both access and refresh tokens
        access_token = create_access_token(identity={
            'id': user_id,
            'email': email
        })
        refresh_token = create_refresh_token(identity={
            'id': user_id,
            'email': email
        })
        
        return jsonify({
            "success": True,
            "token": access_token,
            "refresh_token": refresh_token,
            "name": name
        })
    except Exception as e:
        app.logger.error(f"Login error: {str(e)}")
        return jsonify({"success": False, "error": "Login failed"}), 401

# Add a token refresh endpoint
@app.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """Refresh access token using refresh token"""
    current_user = get_jwt_identity()
    access_token = create_access_token(identity=current_user)
    return jsonify(access_token=access_token)

@app.route("/auth/google", methods=["POST"])
@limiter.exempt
def google_auth():
    try:
        data = request.get_json()
        token = data.get('token')
        
        # Verify the token with clock skew allowance
        idinfo = id_token.verify_oauth2_token(
            token, 
            google_requests.Request(), 
            GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=30
        )
        
        # Get user info
        email = idinfo['email']
        name = idinfo['name']
        google_id = idinfo['sub']
        
        # Check if user exists in Supabase
        user_query = supabase.table('users').select('*').eq('email', email).execute()
        
        is_new_user = not user_query.data
        
        if is_new_user:
            # Create user in Supabase
            try:
                auth_user = supabase.auth.sign_up({
                    "email": email,
                    "password": secrets.token_urlsafe(16)  # Random password for OAuth users
                })
                
                user_id = auth_user.user.id
                
                # Store user data in 'users' table
                supabase.table('users').insert({
                    "id": user_id,
                    "email": email,
                    "name": name,
                    "created_at": datetime.datetime.now().isoformat(),
                    "auth_provider": "google",
                    "google_id": google_id,
                    "plan_tier": "free",
                    "subscription_tier": "free",
                    "subscription_status": "inactive"
                }).execute()
            except Exception as e:
                app.logger.error(f"Error creating new user via Google: {str(e)}")
                # Return specific error for new user that couldn't be created
                return jsonify({
                    "success": False, 
                    "error": "Unable to create account",
                    "is_new_user": True,
                    "email": email,
                    "name": name
                }), 400
        else:
            user_id = user_query.data[0]['id']
            
            # Update last login
            supabase.table('users').update({
                "last_login": datetime.datetime.now().isoformat()
            }).eq('id', user_id).execute()
        
        # Generate JWT token with user ID
        token = jwt.encode({
            'id': user_id,
            'email': email,
            'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7)
        }, JWT_SECRET, algorithm="HS256")
        
        response = jsonify({
            "success": True,
            "token": token,
            "name": name,
            "is_new_user": is_new_user  # Add flag indicating if this is a new user
        })
        
        # Force CORS headers directly on this response
        origin = request.headers.get('Origin')
        if origin in ["https://cardmatcher.net", "https://www.cardmatcher.net", "http://localhost:3000"]:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            response.headers['Cross-Origin-Opener-Policy'] = 'same-origin-allow-popups'
        
        return response
    except Exception as e:
        app.logger.error(f"Google auth error: {str(e)}")
        return jsonify({"success": False, "error": "Authentication failed"}), 401

@app.route("/auth/google", methods=["OPTIONS"])
def google_auth_options():
    response = app.make_response("")
    response.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin', '*'))
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    # Change this to support Google's auth popup
    response.headers.add('Cross-Origin-Opener-Policy', 'same-origin-allow-popups')
    return response

@app.route("/create_link_token", methods=["POST"])
@token_required
def create_link_token(current_user):
    try:
        app.logger.info(f"Creating link token for user {current_user['id']}")
        
        # Create link token with Plaid
        client_user_id = current_user['id']
        
        # Configure Plaid token request
        request_data = {
            'user': {
                'client_user_id': client_user_id
            },
            'client_name': 'Card Matcher',
            'products': CONFIG['PLAID_PRODUCTS'],
            'language': 'en',
            'country_codes': CONFIG['PLAID_COUNTRY_CODES']
        }
        
        # Only add webhook if it exists
        if CONFIG['PLAID_WEBHOOK']:
            request_data['webhook'] = CONFIG['PLAID_WEBHOOK']
        
        app.logger.info(f"Plaid link token request: {request_data}")
        
        # Create the link token
        link_token_response = client.link_token_create(request_data)
        link_token = link_token_response['link_token']
        
        app.logger.info(f"Successfully created link token")
        
        # Return the link token
        return jsonify({
            'success': True,
            'link_token': link_token
        })
    except plaid.ApiException as e:
        app.logger.error(f"Plaid API error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Failed to create link token: {e.body}"
        }), 500
    except Exception as e:
        app.logger.error(f"Error creating link token: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f"Failed to create link token: {str(e)}"
        }), 500

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

@app.route("/get_transactions", methods=["POST"])
def get_transactions():
    try:
        public_token = request.json.get("public_token")
        if not public_token:
            app.logger.error("Missing public_token")
            return jsonify({"error": "Missing public_token"}), 400

        app.logger.info("Processing transaction request...")

        try:
            # Exchange public token for access token
            exchange_response = client.item_public_token_exchange(
                {'public_token': public_token}
            )
            access_token = exchange_response['access_token']
            app.logger.info("Token exchange successful")

            # Get transactions
            start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            
            transactions_response = client.transactions_get(
                transactions_get_request={
                    'access_token': access_token,
                    'start_date': start_date,
                    'end_date': end_date,
                    'options': {
                        'count': 100,
                        'offset': 0
                    }
                }
            )
            
            # Log the raw response for debugging
            app.logger.info("Raw Plaid Response:")
            app.logger.info(f"Accounts: {transactions_response.get('accounts', [])}")
            app.logger.info(f"Item: {transactions_response.get('item', {})}")
            app.logger.info(f"Total Transactions: {len(transactions_response.get('transactions', []))}")
            
            # Log a sample transaction if available
            if transactions_response.get('transactions'):
                sample_tx = transactions_response['transactions'][0]
                app.logger.info("Sample Transaction Structure:")
                try:
                    # Use custom encoder for logging
                    serialized_tx = json.dumps(
                        sample_tx,
                        indent=2,
                        cls=CustomJSONEncoder
                    )
                    app.logger.info(serialized_tx)
                except Exception as e:
                    app.logger.error(f"Error serializing sample transaction: {str(e)}")
            
            # Convert Plaid transactions to dictionaries
            raw_transactions = transactions_response.get('transactions', [])
            transactions = []
            for tx in raw_transactions:
                try:
                    if isinstance(tx, plaid.model.transaction.Transaction):
                        tx_dict = json.loads(
                            json.dumps(tx, cls=CustomJSONEncoder)
                        )
                    else:
                        tx_dict = tx
                    transactions.append(tx_dict)
                except Exception as e:
                    app.logger.error(f"Error processing transaction: {str(e)}")
                    continue

            app.logger.info(f"Successfully processed {len(transactions)} transactions")

            if not transactions:
                app.logger.info("No transactions found, using sample data")
                transactions = get_sample_transactions()
            
            # Analyze transactions
            analysis = analyze_transactions(transactions)
            
            return jsonify({
                "transactions": transactions,
                "analysis": analysis,
                "date_range": {
                    "start": start_date,
                    "end": end_date
                }
            })

        except plaid.ApiException as e:
            app.logger.error(f"Plaid API error: {str(e)}")
            transactions = get_sample_transactions()
            analysis = analyze_transactions(transactions)
            
            return jsonify({
                "transactions": transactions,
                "analysis": analysis,
                "date_range": {
                    "start": start_date,
                    "end": end_date
                },
                "warning": "Using sample data due to API error"
            })

    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    file = request.files["file"]
    df = pd.read_csv(file)
    transactions = df.to_dict(orient="records")
    cashback = calculate_cashback(transactions)
    return jsonify({"transactions": transactions, "cashback": cashback})

@app.route("/compare_cards", methods=["POST"])
@limiter.limit("5 per minute")
def compare_cards():
    try:
        public_token = request.json.get("public_token")
        if not public_token:
            return jsonify({"error": "Missing public_token"}), 400
            
        # Get transactions and analyze spending
        transactions = fetch_plaid_transactions(public_token)
        spending_analysis = analyze_transactions(transactions)
        
        # Get and analyze cards
        credit_cards = fetch_credit_cards()
        card_analysis = []
        
        for card in credit_cards:
            rewards = calculate_card_rewards(transactions, card)
            
            card_analysis.append({
                'card_name': card.get('cardName'),
                'issuer': card.get('cardIssuer'),
                'rewards_earned': rewards['total_rewards'],
                'effective_rate': rewards['effective_rate'],
                'category_breakdown': rewards['category_rewards'],
                'spending_limits': rewards.get('spending_limits', {}),
                'base_earn_type': card.get('baseSpendEarnType'),
                'base_earn_category': card.get('baseSpendEarnCategory'),
                'annual_fee': card.get('annualFee', 0),
                'net_rewards': rewards['total_rewards'] - float(card.get('annualFee', 0)),
                'card_details': card
            })
        
        # Sort by net rewards
        card_analysis.sort(key=lambda x: x['net_rewards'], reverse=True)
        
        return jsonify({
            "recommendations": card_analysis[:3],  # Top 3 cards
            "spending_summary": spending_analysis,
            "analysis_period": {
                "start": (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d'),
                "end": datetime.datetime.now().strftime('%Y-%m-%d')
            }
        })
        
    except Exception as e:
        app.logger.error(f"Error in compare_cards: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/fetch_all_card_rewards", methods=["GET"])
@limiter.limit("5 per minute")
@token_required  # Now requires authentication
def fetch_all_card_rewards(current_user):
    """Fetch Plaid reward details and calculate potential rewards for each card"""
    try:
        # Get transactions (either from Plaid or sample data)
        public_token = request.args.get("public_token")
        if public_token:
            try:
                transactions = fetch_plaid_transactions(public_token)
            except Exception as e:
                app.logger.warning(f"Failed to fetch Plaid transactions: {str(e)}")
                app.logger.info("Using sample transactions instead")
                transactions = get_sample_transactions()
        else:
            app.logger.info("No public token provided, using sample transactions")
            transactions = get_sample_transactions()

        credit_cards = [
            "amex-gold",
            "chase-sapphire-preferred",
            "capital-one-venture-rewards",
            "amex-platinum",
            "citi-premier",
            "capital-one-venture-x",
            "discover-it-cash-back",
            "chase-freedom-unlimited",
            "citi-double-cash",
            "wells-fargo-active-cash",
            "amex-blue-cash-preferred",
            "bofa-premium-rewards",
            "capital-one-savor",
            "citi-custom-cash",
            "us-bank-altitude-go",
            "chase-freedom-flex",
            "ink-business-preferred",
            "amex-delta-reserve",
            "amex-hilton-aspire",
            "capital-one-spark-cash-plus"
        ]
        
        app.logger.info(f"Starting to fetch rewards data for {len(credit_cards)} cards")
        
        # Instead of async code, use synchronous approach
        card_analysis = []
        
        for card_key in credit_cards[:5]:  # Limit to 5 cards for quick testing
            try:
                card_data = fetch_card_details_sync(card_key)
                
                # Handle case where card_data is a list
                if isinstance(card_data, list) and len(card_data) > 0:
                    card_info = card_data[0]  # Take the first item in the list
                else:
                    card_info = card_data
                
                rewards_calculation = calculate_card_potential_rewards(transactions, card_info)
                
                card_analysis.append({
                    'card_key': card_key,
                    'card_name': card_info.get('cardName'),
                    'issuer': card_info.get('cardIssuer'),
                    'potential_rewards': {
                        'total_cashback': rewards_calculation['total_rewards'],
                        'effective_rate': rewards_calculation['effective_rate'],
                        'by_category': rewards_calculation['category_rewards'],
                        'spending_limits': rewards_calculation['spending_limits']
                    },
                    'annual_fee': card_info.get('annualFee', 0),
                    'net_benefit': rewards_calculation['total_rewards'] - float(card_info.get('annualFee', 0)),
                    'card_details': card_info
                })
            except Exception as e:
                app.logger.error(f"Error processing card {card_key}: {str(e)}")
                # Continue with next card
        
        # Sort by net benefit
        card_analysis.sort(key=lambda x: x['net_benefit'], reverse=True)
        
        result = {
            "success": True,
            "analysis_period": {
                "start": (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d'),
                "end": datetime.datetime.now().strftime('%Y-%m-%d')
            },
            "total_transactions": len(transactions),
            "total_spend": sum(abs(float(t.get('amount', 0))) for t in transactions),
            "card_analysis": card_analysis,
            "stats": {
                "total_cards": len(credit_cards),
                "analyzed_cards": len(card_analysis)
            },
            "note": "Using sample transaction data" if not public_token else "Using actual transaction data"
        }
        
        # Store the recommendation in Supabase
        try:
            # First, mark all previous recommendations as not current
            supabase.table('card_recommendations') \
                .update({"is_current": False}) \
                .eq('user_id', current_user['id']) \
                .execute()
            
            # Insert the new recommendation
            supabase.table('card_recommendations').insert({
                "user_id": current_user['id'],
                "created_at": datetime.datetime.now().isoformat(),
                "analysis_data": result,
                "is_current": True
            }).execute()
            
            app.logger.info(f"Stored card recommendations for user {current_user['id']}")
        except Exception as e:
            app.logger.error(f"Failed to store recommendations: {str(e)}")
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"Error in fetch_all_card_rewards: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/get_stored_recommendations", methods=["GET"])
@token_required
def get_stored_recommendations(current_user):
    try:
        # Get current recommendation
        response = supabase.table('card_recommendations') \
            .select('*') \
            .eq('user_id', current_user['id']) \
            .eq('is_current', True) \
            .order('created_at', desc=True) \
            .limit(1) \
            .execute()
        
        if response.data:
            return jsonify({
                "success": True,
                "has_recommendations": True,
                "recommendations": response.data[0]['analysis_data']
            })
        else:
            return jsonify({
                "success": True, 
                "has_recommendations": False,
                "message": "No recommendations found. Connect your bank to get personalized card recommendations."
            })
    except Exception as e:
        app.logger.error(f"Error retrieving recommendations: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/recommendation_history", methods=["GET"])
@token_required
def recommendation_history(current_user):
    try:
        # Get all recommendations for the user
        response = supabase.table('card_recommendations') \
            .select('id, created_at, is_current') \
            .eq('user_id', current_user['id']) \
            .order('created_at', desc=True) \
            .execute()
        
        return jsonify({
            "success": True,
            "history": response.data
        })
    except Exception as e:
        app.logger.error(f"Error retrieving recommendation history: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/recommendation/<recommendation_id>", methods=["GET"])
@token_required
def get_recommendation_by_id(current_user, recommendation_id):
    try:
        # Get the specific recommendation
        response = supabase.table('card_recommendations') \
            .select('*') \
            .eq('id', recommendation_id) \
            .eq('user_id', current_user['id']) \
            .limit(1) \
            .execute()
        
        if response.data:
            return jsonify({
                "success": True,
                "recommendation": response.data[0]['analysis_data']
            })
        else:
            return jsonify({
                "success": False,
                "error": "Recommendation not found"
            }), 404
    except Exception as e:
        app.logger.error(f"Error retrieving recommendation: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/plans', methods=['GET'])
def get_plans():
    """Return all payment plans"""
    return jsonify({
        'success': True,
        'plans': PAYMENT_PLANS
    })

@app.route('/create-checkout-session', methods=['POST'])
@token_required
def create_checkout_session(current_user):
    """Create a Stripe checkout session with idempotency key for safety"""
    try:
        data = request.get_json()
        plan_id = data.get('plan_id')
        
        if plan_id not in ['basic', 'pro']:
            return jsonify({'error': 'Invalid plan selected'}), 400
            
        plan = PAYMENT_PLANS[plan_id]
        
        if not STRIPE_SECRET_KEY:
            return jsonify({'error': 'Stripe is not configured'}), 500

        # Generate a unique idempotency key based on user and plan
        idempotency_key = f"{current_user['id']}_{plan_id}_{datetime.datetime.now().strftime('%Y%m%d')}"
        
        # Get customer ID or create one with idempotency
        user_data = supabase.table('users').select('stripe_customer_id').eq('id', current_user['id']).execute()
        
        customer_id = None
        customer_exists_in_stripe = False
        
        if user_data.data and user_data.data[0].get('stripe_customer_id'):
            stored_customer_id = user_data.data[0]['stripe_customer_id']
            
            # Check if the customer still exists in Stripe
            try:
                stripe.Customer.retrieve(stored_customer_id)
                customer_id = stored_customer_id
                customer_exists_in_stripe = True
            except stripe.error.InvalidRequestError as e:
                app.logger.warning(f"Customer {stored_customer_id} not found in Stripe: {str(e)}")
        
        # Create a new customer if needed (with idempotency)
        if not customer_exists_in_stripe:
            customer = stripe.Customer.create(
                email=current_user.get('email'),
                name=current_user.get('name', 'Card Matcher User'),
                metadata={'user_id': current_user['id']},
                idempotency_key=f"customer_{current_user['id']}"  # Prevent duplicate customer creation
            )
            customer_id = customer.id
            
            supabase.table('users').update({'stripe_customer_id': customer_id}).eq('id', current_user['id']).execute()
            app.logger.info(f"Created new Stripe customer {customer_id} for user {current_user['id']}")

        # Create checkout session with idempotency key
        checkout_session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=['card'],
            line_items=[{
                'price': plan['stripe_price_id'],
                'quantity': 1,
            }],
            mode='payment',
            success_url=request.headers.get('Origin', '') + '/payment-success.html?payment=success&plan=' + plan_id,
            cancel_url=request.headers.get('Origin', '') + '/pricing.html?payment=canceled',
            metadata={
                'user_id': current_user['id'],
                'plan_id': plan_id
            },
            idempotency_key=idempotency_key  # Prevents accidental duplicate charges
        )
        
        return jsonify({'success': True, 'session_id': checkout_session.id})
        
    except Exception as e:
        app.logger.error(f"Error creating checkout session: {str(e)}")
        return jsonify({'error': "Payment processing error"}), 500  # Sanitized error

@app.route('/customer-portal', methods=['POST'])
@token_required
def customer_portal(current_user):
    """Create a customer portal session for managing subscription"""
    try:
        # Get customer ID
        user_data = supabase.table('users').select('stripe_customer_id').eq('id', current_user['id']).execute()
        
        if not user_data.data or not user_data.data[0].get('stripe_customer_id'):
            return jsonify({'error': 'No subscription found'}), 404
            
        customer_id = user_data.data[0]['stripe_customer_id']
        
        # Create portal session
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=request.headers.get('Origin', '') + '/dashboard.html',
        )
        
        return jsonify({'success': True, 'url': session.url})
        
    except Exception as e:
        app.logger.error(f"Error creating portal session: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/webhook', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhooks for one-time payments"""
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    
    try:
        if not STRIPE_WEBHOOK_SECRET:
            structured_logger.error("Webhook secret not configured")
            return jsonify({'error': "Server configuration error"}), 500
            
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
        
        # Log the webhook type
        structured_logger.info(
            "Received webhook", 
            event_type=event['type'],
            event_id=event['id']
        )
        
        # Handle the event
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            
            # Only handle payment mode sessions
            if session.get('mode') == 'payment':
                record_payment(session)
                
        return jsonify({'status': 'success'})
        
    except stripe.error.SignatureVerificationError as e:
        # This is a specific error we want to log differently
        structured_logger.error(
            "Webhook signature verification failed",
            error=str(e)
        )
        return jsonify({'error': "Invalid signature"}), 401
        
    except Exception as e:
        structured_logger.error(
            "Webhook processing error",
            error=str(e),
            error_type=type(e).__name__,
            traceback=traceback.format_exc()
        )
        return jsonify({'error': "Webhook processing failed"}), 400
def record_payment(session):
    """Record successful one-time payment"""
    # Get data from session
    customer_id = session.get('customer')
    metadata = session.get('metadata', {})
    plan_id = metadata.get('plan_id')
    user_id = metadata.get('user_id')
    
    if not customer_id or not plan_id or not user_id:
        app.logger.error(f"Missing required data in session: {session.id}")
        return
    
    # Insert payment record
    try:
        # Record the payment
        payment_id = uuid.uuid4()
        
        supabase.table('payments').insert({
            "id": str(payment_id),
            "user_id": user_id,
            "stripe_session_id": session.id,
            "stripe_payment_intent": session.get('payment_intent'),
            "plan_id": plan_id,
            "amount": session.get('amount_total') / 100.0,  # Convert from cents
            "created_at": datetime.datetime.now().isoformat(),
            "status": "completed"
        }).execute()
        
        # Update the user's plan
        supabase.table('users').update({
            'plan_tier': plan_id,
            'plan_purchased_at': datetime.datetime.now().isoformat(),
            'payment_id': str(payment_id)
        }).eq('id', user_id).execute()
        
        app.logger.info(f"Payment recorded for user {user_id}, plan: {plan_id}")
        
    except Exception as e:
        app.logger.error(f"Error recording payment: {str(e)}")

@app.route('/subscription', methods=['GET'])
@token_required
def get_subscription(current_user):
    """Get the current user's subscription details"""
    try:
        # Get subscription from Supabase
        user_data = supabase.table('users').select('subscription_tier,subscription_status,subscription_current_period_end').eq('id', current_user['id']).execute()
        
        if not user_data.data:
            return jsonify({'error': 'User not found'}), 404
            
        user = user_data.data[0]
        
        # Default to free tier if no subscription
        subscription_tier = user.get('subscription_tier', 'free')
        subscription_status = user.get('subscription_status', 'inactive')
        
        # Get plan details
        plan = PAYMENT_PLANS.get(subscription_tier, PAYMENT_PLANS['free'])
        
        return jsonify({
            'success': True,
            'subscription': {
                'tier': subscription_tier,
                'status': subscription_status,
                'current_period_end': user.get('subscription_current_period_end'),
                'plan': plan
            }
        })
        
    except Exception as e:
        app.logger.error(f"Error retrieving subscription: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add health check endpoints
@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/health/detailed', methods=['GET'])
@limiter.exempt
def detailed_health_check():
    """Detailed health check that tests all components"""
    health_data = {
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'components': {
            'app': 'healthy',
            'database': 'unknown',
            'stripe': 'unknown'
        },
        'checks': []
    }
    
    # Check database connection
    try:
        start_time = datetime.datetime.now()
        db_response = supabase.table('_health').select('*').limit(1).execute()
        end_time = datetime.datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        health_data['components']['database'] = 'healthy'
        health_data['checks'].append({
            'component': 'database',
            'status': 'healthy',
            'response_time_seconds': response_time
        })
    except Exception as e:
        health_data['status'] = 'degraded'
        health_data['components']['database'] = 'unhealthy'
        health_data['checks'].append({
            'component': 'database',
            'status': 'unhealthy',
            'error': str(e)
        })
    
    # Check Stripe connection if configured
    if STRIPE_SECRET_KEY:
        try:
            start_time = datetime.datetime.now()
            stripe.Account.retrieve()
            end_time = datetime.datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            health_data['components']['stripe'] = 'healthy'
            health_data['checks'].append({
                'component': 'stripe',
                'status': 'healthy',
                'response_time_seconds': response_time
            })
        except Exception as e:
            health_data['status'] = 'degraded'
            health_data['components']['stripe'] = 'unhealthy'
            health_data['checks'].append({
                'component': 'stripe',
                'status': 'unhealthy',
                'error': str(e)
            })
    
    # Check if any component is unhealthy
    if any(value == 'unhealthy' for value in health_data['components'].values()):
        health_data['status'] = 'degraded'
        return jsonify(health_data), 503  # Service Unavailable
    
    return jsonify(health_data)

# Add suspicious activity monitoring
@app.before_request
def monitor_suspicious_activity():
    """Monitor for suspicious activity in requests"""
    # Check for suspicious patterns in user-agent, referrer, or IP
    user_agent = request.headers.get('User-Agent', '')
    referrer = request.headers.get('Referer', '')
    client_ip = request.remote_addr
    
    # Define suspicious patterns
    suspicious_ua_patterns = ['sqlmap', 'nikto', 'nmap', 'burpsuite', 'OWASP ZAP']
    suspicious_paths = ['/wp-login', '/wp-admin', '/admin', '/phpmyadmin']
    
    # Check for suspicious activity
    is_suspicious = False
    reason = None
    
    # Check user agent
    for pattern in suspicious_ua_patterns:
        if pattern.lower() in user_agent.lower():
            is_suspicious = True
            reason = f"Suspicious user agent: {user_agent}"
            break
    
    # Check request path
    if not is_suspicious and request.path:
        for path in suspicious_paths:
            if path in request.path:
                is_suspicious = True
                reason = f"Suspicious path requested: {request.path}"
                break
    
    # Log suspicious activity
    if is_suspicious:
        structured_logger.warning(
            "Suspicious activity detected",
            reason=reason,
            client_ip=client_ip,
            user_agent=user_agent,
            referrer=referrer,
            path=request.path,
            method=request.method
        )
        
        # For highly suspicious activity, you might want to add the IP to a blocklist
        # or return a 403 Forbidden response


