from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import redis
import plaid
from plaid.api import plaid_api
from dotenv import load_dotenv
from flask_talisman import Talisman
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

load_dotenv()

# Add Plaid configuration
PLAID_CLIENT_ID = os.environ.get('PLAID_CLIENT_ID')
PLAID_SECRET = os.environ.get('PLAID_SECRET')
PLAID_ENV = os.environ.get('PLAID_ENV', 'sandbox')

# Add RapidAPI configuration
REWARDS_CC_API_KEY = os.environ.get('RAPIDAPI_KEY')
REWARDS_CC_API_HOST = os.environ.get('RAPIDAPI_HOST')
REWARDS_CC_BASE_URL = os.environ.get('RAPIDAPI_BASE_URL')

# Validate required environment variables
required_env_vars = {
    'PLAID_CLIENT_ID': PLAID_CLIENT_ID,
    'PLAID_SECRET': PLAID_SECRET,
    'REWARDS_CC_API_KEY': REWARDS_CC_API_KEY,
    'REWARDS_CC_API_HOST': REWARDS_CC_API_HOST,
    'REWARDS_CC_BASE_URL': REWARDS_CC_BASE_URL
}

missing_vars = [var for var, value in required_env_vars.items() if not value]
if missing_vars:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")


# Configure Plaid client
configuration = plaid.Configuration(
    host=plaid.Environment.Sandbox,
    api_key={
        'clientId': PLAID_CLIENT_ID,
        'secret': PLAID_SECRET,
        'plaidVersion': '2020-09-14'  # Add API version
    }
)

api_client = plaid.ApiClient(configuration)
client = plaid_api.PlaidApi(api_client)

app = Flask(__name__)

# Add this CORS configuration before any routes
CORS(app, resources={
    r"/*": {
        "origins": [os.environ.get('CORS_ORIGIN', 'https://your-frontend-app.herokuapp.com')],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Disable SSL requirement for local development
app.config['TALISMAN_ENABLED'] = False

# Secure configurations
app.config.update(
    SECRET_KEY=os.environ.get('FLASK_SECRET_KEY', secrets.token_urlsafe(32)),
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

# Configure Redis and rate limiting
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=redis_url,
    storage_options={},
    default_limits=["200 per day", "50 per hour"]
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Add this function after your other helper functions
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
                          for k, v in running_totals.items()}
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

@app.route("/create_link_token", methods=["POST"])
def create_link_token():
    try:
        request_config = {
            'user': {'client_user_id': 'user-' + str(datetime.datetime.now().timestamp())},
            'client_name': "Card Matcher",
            'products': ["transactions"],
            'country_codes': ["US"],
            'language': "en"
        }
        
        response = client.link_token_create(request_config)
        return jsonify({"link_token": response['link_token']})
    except plaid.ApiException as e:
        app.logger.error(f'Error creating link token: {str(e)}')
        return jsonify({"error": str(e)}), 500

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

# Add after Flask app initialization
app.json_encoder = CustomJSONEncoder

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

@app.route("/fetch_all_card_rewards", methods=["GET"])
@limiter.limit("5 per minute")
async def fetch_all_card_rewards():
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
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            batch_size = 2
            card_analysis = []
            
            for i in range(0, len(credit_cards), batch_size):
                batch = credit_cards[i:i + batch_size]
                tasks = [fetch_card_details(card, client) for card in batch]
                results = await asyncio.gather(*tasks)
                
                # Process results
                for result in results:
                    for card_key, card_data in result.items():
                        if card_data:
                            # Handle list response from API
                            if isinstance(card_data, list) and len(card_data) > 0:
                                card_info = card_data[0]  # Take first item from list
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
                
                await asyncio.sleep(1.5)  # Rate limit delay
            
            # Sort by net benefit
            card_analysis.sort(key=lambda x: x['net_benefit'], reverse=True)
            
            return jsonify({
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
            })
    
    except Exception as e:
        app.logger.error(f"Error in fetch_all_card_rewards: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

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



from hypercorn.config import Config
from hypercorn.asyncio import serve

if __name__ == "__main__":
    import asyncio
    
    config = Config()
    config.bind = ["0.0.0.0:5001"]
    config.use_reloader = True
    
    app.logger.info("Starting Hypercorn server...")
    asyncio.run(serve(app, config))

