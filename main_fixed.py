import os
import json
import logging
import pymongo
import urllib.parse
import spacy
import math
import random
import asyncio
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic

# Define DummyNLP in global scope for fallback
class DummyNLP:
    def __init__(self):
        self.name = "DummyNLP-Fallback"
        self.vocab = type('obj', (object,), {
            'vectors': type('obj', (object,), {'n_keys': 0})
        })
        
    def __call__(self, text):
        class DummyDoc:
            def __init__(self, text):
                self.text = text
                self.vector = [0] * 300  # Empty vector
                self.vector_norm = 0
                
            def similarity(self, other):
                # Fallback similarity using Jaccard index on word overlap
                words1 = set(self.text.lower().split())
                words2 = set(other.text.lower().split())
                
                if not words1 or not words2:
                    return 0
                    
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                
                return len(intersection) / len(union)
        
        return DummyDoc(text)

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ✅ Securely Connect to MongoDB
password = os.environ.get("MONGO_PASSWORD", "master2002_B*")  # Fallback for development
encoded_password = urllib.parse.quote_plus(password)

MONGO_URI = f"mongodb+srv://abdelrahman:{encoded_password}@cluster0.goxvb.mongodb.net/travel_app?retryWrites=true&w=majority&appName=Cluster0"

def connect_mongo(uri, retries=3, retry_delay=2):
    """Attempts to connect to MongoDB with improved retry logic."""
    for attempt in range(retries):
        try:
            client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.server_info()  # Force connection check
            logger.info("✅ MongoDB connection established!")
            return client
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"❌ MongoDB connection failed (Attempt {attempt + 1}/{retries}): {e}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"❌ MongoDB connection failed after {retries} attempts: {e}")
                raise Exception(f"❌ MongoDB connection failed after {retries} attempts: {e}")

client = connect_mongo(MONGO_URI)
db = client["travel_app"]

# Define Collections
users_collection = db["users"]
places_collection = db["places"]
interactions_collection = db["interactions"]
search_queries_collection = db["search_queries"]
travel_preferences_collection = db["user_travel_preferences"]
recommendations_cache_collection = db["recommendations_cache"]
shown_places_collection = db["shown_places"]
roadmaps_collection = db["roadmaps"]  # For roadmaps
cache_locks_collection = db["cache_locks"]  # New collection for tracking locks

# --- CREATE TTL INDEXES ---
# These indexes automatically remove documents after a specified time period

# TTL index for roadmaps (expires after 24 hours)
try:
    roadmaps_collection.create_index(
        [("created_at", pymongo.ASCENDING)],
        expireAfterSeconds=86400  # 24 hours
    )
    logger.info("✅ Created TTL index on roadmaps collection")
except Exception as e:
    logger.error(f"❌ Error creating TTL index on roadmaps collection: {e}")

# TTL index for recommendations cache (expires after 6 hours)
try:
    recommendations_cache_collection.create_index(
        [("timestamp", pymongo.ASCENDING)],
        expireAfterSeconds=21600  # 6 hours
    )
    logger.info("✅ Created TTL index on recommendations_cache collection")
except Exception as e:
    logger.error(f"❌ Error creating TTL index on recommendations_cache collection: {e}")

# TTL index for shown places (expires after 6 hours)
try:
    shown_places_collection.create_index(
        [("timestamp", pymongo.ASCENDING)],
        expireAfterSeconds=21600  # 6 hours
    )
    logger.info("✅ Created TTL index on shown_places collection")
except Exception as e:
    logger.error(f"❌ Error creating TTL index on shown_places collection: {e}")

# TTL index for cache locks (expires after 10 minutes)
try:
    cache_locks_collection.create_index(
        [("timestamp", pymongo.ASCENDING)],
        expireAfterSeconds=600  # 10 minutes (safety cleanup for stale locks)
    )
    logger.info("✅ Created TTL index on cache_locks collection")
except Exception as e:
    logger.error(f"❌ Error creating TTL index on cache_locks collection: {e}")

# Create index on user_id field for better query performance
for collection_name in ["recommendations_cache", "shown_places", "roadmaps", "cache_locks"]:
    try:
        db[collection_name].create_index([("user_id", pymongo.ASCENDING)])
        logger.info(f"✅ Created user_id index on {collection_name} collection")
    except Exception as e:
        logger.error(f"❌ Error creating index on {collection_name}: {e}")

# --- Initialize spaCy model ---
def load_spacy_model(model="en_core_web_md", retries=2):  # Use medium model by default
    """Attempts to load the spaCy model with better vector checking."""
    logger.info(f"🔄 Attempting to load spaCy model: {model}")
    
    for attempt in range(retries):
        try:
            nlp = spacy.load(model)
            
            # Verify that the model has word vectors
            test_doc = nlp("travel")
            has_vectors = nlp.vocab.vectors.n_keys > 0 and test_doc.vector_norm > 0
            
            if has_vectors:
                logger.info(f"✅ Successfully loaded spaCy model: {model} WITH WORD VECTORS")
                return nlp
            else:
                logger.warning(f"⚠️ Model {model} loaded but NO WORD VECTORS found!")
                
                # If this is the 'md' model and it doesn't have vectors, try 'sm' model
                if model == "en_core_web_md" and attempt == 0:
                    logger.info("⚠️ Medium model doesn't have vectors. Attempting to download vectors...")
                    try:
                        import subprocess
                        result = subprocess.run([sys.executable, "-m", "spacy", "download", model], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            logger.info(f"✅ Successfully downloaded model: {model}")
                            continue  # Try loading again
                    except Exception as download_err:
                        logger.error(f"❌ Failed to download model: {download_err}")
                
                # If we can't fix the current model, fall back to the small model
                if model != "en_core_web_sm":
                    logger.info("🔄 Falling back to small model...")
                    return load_spacy_model("en_core_web_sm", 1)
        except Exception as e:
            logger.error(f"❌ Error loading NLP model (Attempt {attempt + 1}/{retries}): {e}")
            try:
                logger.info(f"📥 Downloading spaCy model: {model}")
                import subprocess
                result = subprocess.run([sys.executable, "-m", "spacy", "download", model], 
                                       capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"✅ Successfully downloaded model: {model}")
                else:
                    logger.error(f"❌ Failed to download model: {result.stderr}")
            except Exception as download_err:
                logger.error(f"❌ Failed to download model: {download_err}")

    # Return dummy NLP object from the global class
    logger.warning("⚠️ CRITICAL: Using dummy NLP model as fallback! Semantic search will use word overlap instead.")
    return DummyNLP()

# Try to load the model with word vectors
nlp = load_spacy_model()

# Check if model has word vectors and log clearly
test_text = "travel"
test_doc = nlp(test_text)
has_vectors = hasattr(test_doc, 'vector_norm') and test_doc.vector_norm > 0

if has_vectors:
    logger.info("✅ SUCCESS: NLP Model loaded with WORD VECTORS - semantic search will work properly")
else:
    logger.warning("⚠️ WARNING: NLP Model doesn't have word vectors - semantic search will use fallback algorithm")
# Initialize FastAPI app
app = FastAPI(
    title="Travel API",
    description="API for travel recommendations and roadmaps",
    version="2.0.0"  # Updated version
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- PART 2: MODELS AND SHARED UTILITY FUNCTIONS ---

# --- Pydantic Models ---
class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: Optional[int] = 10

class RoadmapRequest(BaseModel):
    user_id: str

class SearchRequest(BaseModel):
    user_id: str
    query: str
    limit: Optional[int] = 10

# --- Shared Utility Functions ---

def get_user_data(user_id):
    """Get complete user data including preferences"""
    user = users_collection.find_one({"_id": user_id})
    
    if not user:
        logger.warning(f"User {user_id} not found")
        return None
        
    return user

def get_user_preferences(user_id):
    """Get user general preferences (categories & tags only)"""
    user = get_user_data(user_id)
    
    if not user:
        return None
        
    # Handle the nested preferences structure
    preferences = user.get("preferences", {})
    
    return {
        "preferred_categories": preferences.get("categories", []),
        "preferred_tags": preferences.get("tags", []),
    }

def get_user_travel_preferences(user_id):
    """Get user travel-specific preferences, including budget"""
    travel_prefs = travel_preferences_collection.find_one(
        {"user_id": user_id}
    )
    
    if not travel_prefs:
        logger.warning(f"No travel preferences found for user {user_id}")
        return None
        
    return {
        "destinations": travel_prefs.get("destinations", []),
        "travel_dates": travel_prefs.get("travel_dates", ""),
        "accessibility_needs": travel_prefs.get("accessibility_needs", []),
        "budget": travel_prefs.get("budget", "medium"),  # Default to 'medium' if missing
        "group_type": travel_prefs.get("group_type", "")  # Added group_type
    }

def compute_text_similarity(text1, text2):
    """
    Compute similarity between two text strings.
    Uses spaCy word vectors if available, falls back to word overlap otherwise.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0
        
    try:
        # Try using spaCy word vectors
        doc1 = nlp(text1.lower())
        doc2 = nlp(text2.lower())
        
        # Check if vectors are available (not all models have vectors)
        if hasattr(doc1, 'vector_norm') and doc1.vector_norm > 0 and hasattr(doc2, 'vector_norm') and doc2.vector_norm > 0:
            similarity = doc1.similarity(doc2)
            logger.debug(f"Vector similarity between '{text1}' and '{text2}': {similarity:.2f}")
            return similarity
        
        # Fall back to basic word overlap if vectors aren't available
        words1 = set(word.lower() for word in text1.split())
        words2 = set(word.lower() for word in text2.split())
        
        if not words1 or not words2:
            return 0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0
        logger.debug(f"Jaccard similarity between '{text1}' and '{text2}': {jaccard_similarity:.2f}")
        return jaccard_similarity
        
    except Exception as e:
        logger.error(f"Error computing text similarity: {str(e)}")
        
        # Emergency fallback
        return 0

def parse_travel_dates(travel_dates_str):
    """
    Parse the travel_dates string to extract month information.
    
    Args:
        travel_dates_str: String containing travel dates (e.g. "March 2025", "now", "August 2025")
        
    Returns:
        String containing the month name or None if not parseable
    """
    if not travel_dates_str:
        return None
        
    # If user selected "now", return current month
    if travel_dates_str.lower() == "now":
        return datetime.now().strftime("%B")  # Returns month name like "March"
        
    # Try to parse the string as a date
    try:
        # Assume format is like "March 2025" or similar
        date_parts = travel_dates_str.split()
        if len(date_parts) >= 1:
            # First part should be the month name
            month_name = date_parts[0].capitalize()
            # Check if it's a valid month name
            valid_months = [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ]
            if month_name in valid_months:
                return month_name
    except Exception as e:
        logger.error(f"Error parsing travel dates: {e}")
        
    # Return current month as fallback
    logger.warning(f"Could not parse travel dates '{travel_dates_str}', using current month")
    return datetime.now().strftime("%B")

def apply_time_decay(weight, interaction_time):
    """
    Apply time-based decay to an interaction weight.
    
    Args:
        weight: Base weight for the interaction
        interaction_time: Timestamp of the interaction
        
    Returns:
        Adjusted weight after time decay
    """
    # Get current time consistently as timezone-naive
    current_date = datetime.now().replace(tzinfo=None).date()
    
    # Handle string timestamps
    if isinstance(interaction_time, str):
        try:
            # Try to parse string timestamp and remove timezone
            timestamp = interaction_time.split("T")[0]  # Take just the date part
            interaction_date = datetime.strptime(timestamp, "%Y-%m-%d").date()
        except Exception as e:
            logger.error(f"Error parsing interaction timestamp: {e}")
            return weight  # Return original weight on error
    # Handle datetime objects
    elif hasattr(interaction_time, 'date'):
        try:
            interaction_date = interaction_time.replace(tzinfo=None).date()
        except Exception as e:
            logger.error(f"Error converting interaction time to date: {e}")
            return weight  # Return original weight on error
    else:
        # Fallback to current date
        logger.warning(f"Invalid interaction_time format: {type(interaction_time)}")
        return weight
        
    # Calculate days between dates
    days_ago = max(0, (current_date - interaction_date).days)
    decay = math.exp(-days_ago / 30)  # Exponential decay over 30 days
    
    return weight * decay

def get_numeric_value(obj, key, default=0):
    """
    Safely extract numeric values from MongoDB documents.
    
    Args:
        obj: MongoDB document or dictionary
        key: Key to extract
        default: Default value if key not found or value not numeric
        
    Returns:
        Numeric value (float or int)
    """
    if not obj or not isinstance(obj, dict):
        return default
        
    value = obj.get(key, default)
    
    # Handle MongoDB numeric types
    if isinstance(value, dict):
        if "$numberDouble" in value:
            return float(value["$numberDouble"])
        elif "$numberInt" in value:
            return int(value["$numberInt"])
        elif "$numberLong" in value:
            return int(value["$numberLong"])
            
    # Try to convert to float if it's a string
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
            
    # Return the value directly if it's already a number
    if isinstance(value, (int, float)):
        return value
        
    return default

def get_user_cached_recommendations(user_id):
    """
    Get all cached recommendation entries for a user, sorted by sequence.
    
    Args:
        user_id: User ID to get cache for
        
    Returns:
        List of cached recommendation entries sorted by sequence
    """
    try:
        # Find all cache entries for this user, sorted by sequence
        return list(recommendations_cache_collection.find(
            {"user_id": user_id, "_id": {"$ne": f"cache_lock_{user_id}"}}
        ).sort("sequence", 1))
    except Exception as e:
        logger.error(f"Error retrieving cached recommendations: {e}")
        return []

def clear_user_cache(user_id):
    """
    Clear all cached recommendations for a user.
    
    Args:
        user_id: User ID to clear cache for
        
    Returns:
        Number of entries deleted
    """
    try:
        result = recommendations_cache_collection.delete_many({"user_id": user_id})
        return result.deleted_count
    except Exception as e:
        logger.error(f"Error clearing user cache: {e}")
        return 0

def store_cache_entry(user_id, recommendations, sequence):
    """
    Store recommendations in cache with the given sequence number.
    
    Args:
        user_id: User ID
        recommendations: Recommendations data
        sequence: Sequence number for this cache entry
        
    Returns:
        True if successful, False otherwise
    """
    try:
        recommendations_cache_collection.insert_one({
            "user_id": user_id,
            "sequence": sequence,
            "recommendations": recommendations,
            "timestamp": datetime.now()
        })
        return True
    except Exception as e:
        logger.error(f"Error storing cache entry: {e}")
        return False
# --- PART 3: RECOMMENDATION ALGORITHM FUNCTIONS ---

def get_candidate_places(user_preferences, user_id, size=30):
    """
    Get candidate places for recommendations based on user preferences.
    Enhanced with improved semantic search for matching places to user preferences.
    
    Args:
        user_preferences: Dictionary containing user information with preferences or user_id
        user_id: User ID for fetching search history and interactions
        size: Maximum number of candidate places to return
        
    Returns:
        List of candidate places
    """
    logger.info(f"Finding candidate places for user {user_id}")
    
    # FIXED: Make sure we have the full user document with preferences
    # If user_preferences doesn't have preferences, try to get the full user document
    if not isinstance(user_preferences, dict) or "preferences" not in user_preferences:
        # Try to fetch the user document from the database
        try:
            # Assuming users_collection is available in this scope
            user_doc = users_collection.find_one({"_id": user_id})
            if user_doc and "preferences" in user_doc:
                user_preferences = user_doc
                logger.info(f"Fetched user document from database for {user_id}")
            else:
                logger.warning(f"Could not find user preferences in database for {user_id}")
        except Exception as e:
            logger.error(f"Error fetching user from database: {e}")
    
    # Extract preferences from the user document
    if isinstance(user_preferences, dict) and "preferences" in user_preferences:
        preferences_obj = user_preferences.get("preferences", {})
        preferred_categories = preferences_obj.get("categories", [])
        preferred_tags = preferences_obj.get("tags", [])
    else:
        preferred_categories = []
        preferred_tags = []
    
    # Log preferences with the correct structure
    logger.info(f"User preferences - Categories: {preferred_categories}, Tags: {preferred_tags}")
    
    # If no preferences, return popular places
    if not preferred_categories and not preferred_tags:
        logger.warning(f"No user preferences found for user {user_id}, returning popular places")
        return list(places_collection.find().sort([("average_rating", -1)]).limit(size))
    
    # --- PART 1: VERY AGGRESSIVE FUZZY MATCHING ---
    all_places = list(places_collection.find())
    scored_places = []
    
    # Convert preferences to lowercase for comparison
    preferred_categories_lower = [cat.lower() for cat in preferred_categories if cat]
    preferred_tags_lower = [tag.lower() for tag in preferred_tags if tag]
    
    # Extract words from categories and tags for partial matching
    category_words = set()
    for cat in preferred_categories_lower:
        category_words.update(cat.split())
    
    tag_words = set()
    for tag in preferred_tags_lower:
        tag_words.update(tag.split())
    
    for place in all_places:
        score = 0
        place_name = place.get("name", "")
        place_category = place.get("category", "").lower() if place.get("category") else ""
        place_tags = [tag.lower() for tag in place.get("tags", [])] if isinstance(place.get("tags"), list) else []
        place_description = place.get("description", "").lower()
        
        # 1. Direct category match (highest weight)
        if place_category and any(cat == place_category for cat in preferred_categories_lower):
            score += 1.0
        
        # 2. Partial category match
        elif place_category:
            # Category contains or is contained by any preferred category
            if any(cat in place_category or place_category in cat for cat in preferred_categories_lower):
                score += 0.7
            # Word-level matching
            elif any(word in place_category for word in category_words):
                score += 0.5
        
        # 3. Tag matching - use both exact and partial
        if place_tags:
            # Direct tag matches
            exact_matches = sum(1 for tag in place_tags if tag in preferred_tags_lower)
            if exact_matches > 0:
                score += 0.8 * min(1.0, exact_matches / len(preferred_tags_lower))
            
            # Partial tag matches - check if any place tag contains or is contained by a preferred tag
            partial_matches = sum(1 for tag in place_tags 
                                if any(pref in tag or tag in pref 
                                      for pref in preferred_tags_lower))
            if partial_matches > 0:
                score += 0.5 * min(1.0, partial_matches / len(preferred_tags_lower))
            
            # Word-level tag matching
            word_matches = sum(1 for tag in place_tags 
                             if any(word in tag for word in tag_words))
            if word_matches > 0:
                score += 0.3 * min(1.0, word_matches / len(tag_words))

        
        # 4. Check description for keywords (bonus match)
        if place_description:
            cat_matches = sum(1 for cat in preferred_categories_lower if cat in place_description)
            tag_matches = sum(1 for tag in preferred_tags_lower if tag in place_description)
            
            if cat_matches > 0 or tag_matches > 0:
                score += 0.2  # Small bonus for description matches
        
        # 5. EXTRA AGGRESSIVE: Give every place at least a minimal score to ensure some matches
        if score == 0:
            score = 0.01
        
        scored_places.append((place, score))
    
    # Sort by score descending
    scored_places.sort(key=lambda x: x[1], reverse=True)
    category_tag_places = [place for place, score in scored_places if score > 0.01]  # Filter minimal scores
    
    # Limit to ensure we don't have too many low-quality matches
    category_tag_places = category_tag_places[:min(len(category_tag_places), size)]
    
    logger.info(f"Found {len(category_tag_places)} places with direct category/tag matching for user {user_id}")
    
    # Continue with the rest of the function (semantic search)...
    # --- PART 2: SEMANTIC SEARCH BASED ON RECENT QUERIES (40%) ---
    semantic_places = []
    
    # Check if NLP is available with more robust verification
    nlp_available = False
    fallback_level = 0
    
    try:
        # Level 1: Check if nlp exists and has vectors
        if nlp and not isinstance(nlp, DummyNLP):
            test_doc = nlp("test")
            if hasattr(test_doc, 'vector_norm') and test_doc.vector_norm > 0:
                nlp_available = True
                logger.info("Using full spaCy NLP with word vectors")
            else:
                fallback_level = 1
                logger.warning("spaCy available but word vectors missing, using fallback level 1")
        else:
            fallback_level = 2
            logger.warning("spaCy not available, using fallback level 2")
    except Exception as e:
        fallback_level = 3
        logger.error(f"Error checking NLP availability: {e}, using fallback level 3")
    
    try:
        # Fetch recent search queries for this user
        search_queries = list(search_queries_collection.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(5))
        
        # Extract keywords from search queries with appropriate fallbacks
        search_keywords = set()

        for query_doc in search_queries:
            # Use existing keywords if available
            if "keywords" in query_doc and query_doc["keywords"]:
                for keyword in query_doc["keywords"]:
                    if keyword and len(keyword) > 2:
                        search_keywords.add(keyword.lower())
            else:
                query_text = query_doc.get("query", "")
                if not query_text:
                    continue
                
                # Different keyword extraction based on fallback level
                if fallback_level == 0:
                    # Full NLP with POS tagging
                    doc = nlp(query_text.lower())
                    for token in doc:
                        if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop and len(token.text) > 2:
                            search_keywords.add(token.text)
                elif fallback_level == 1:
                    # spaCy available but no vectors - use basic POS if available, otherwise tokenize
                    try:
                        doc = nlp(query_text.lower())
                        for token in doc:
                            if not token.is_stop and len(token.text) > 2:
                                search_keywords.add(token.text)
                    except:
                        # Tokenize and filter stopwords
                        stopwords = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
                        for word in query_text.lower().split():
                            if word not in stopwords and len(word) > 2:
                                search_keywords.add(word)
                else:
                    # Basic tokenization for levels 2 and 3
                    stopwords = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
                    for word in query_text.lower().split():
                        if word not in stopwords and len(word) > 2:
                            search_keywords.add(word)
        
        search_keywords_list = list(search_keywords)  # Convert to list for consistent ordering
        logger.info(f"Extracted {len(search_keywords_list)} search keywords for user {user_id}")
        
        if search_keywords_list:
            # Calculate semantic similarity scores based on fallback level
            scored_semantic_places = []
            
            # Fallback level determines matching approach
            if fallback_level == 0:
                # Full semantic matching with spaCy word vectors
                for place in all_places:
                    place_id = place["_id"]

                    # Skip if place is already highly ranked in category/tag matches
                    if place in category_tag_places[:int(size * 0.3)]:
                        continue
                    
                    # Initialize scoring components
                    tag_similarity = 0.0
                    tag_match_count = 0
                    description_similarity = 0.0
                    description_match_count = 0
                    
                    # Get place data
                    tags = place.get("tags", [])
                    description = place.get("description", "")
                    
                    # Process tags with nlp
                    for keyword in search_keywords_list:
                        keyword_doc = nlp(keyword.lower())
                        
                        # Check each tag for similarity
                        for tag in tags:
                            try:
                                tag_doc = nlp(tag.lower())
                                similarity = keyword_doc.similarity(tag_doc)
                                
                                # Count significant matches
                                if similarity > 0.6:  # Threshold for semantic match
                                    tag_similarity += similarity
                                    tag_match_count += 1
                            except Exception as e:
                                continue  # Skip this tag if error
                        
                        # Process description with nlp
                        if description:
                            try:
                                # Process full description
                                desc_doc = nlp(description.lower())
                                
                                # Check semantic similarity
                                similarity = keyword_doc.similarity(desc_doc)
                                
                                # Check exact keyword match in description (bonus)
                                if keyword.lower() in description.lower():
                                    description_similarity += max(similarity, 0.7)  # At least 0.7 for exact match
                                    description_match_count += 1
                                elif similarity > 0.5:  # Lower threshold for description
                                    description_similarity += similarity
                                    description_match_count += 1
                            except Exception as e:
                                logger.debug(f"Error comparing description: {str(e)}")

                    # Calculate final semantic score
                    semantic_score = 0.0
                    
                    # Tag component (60% weight)
                    tag_component = 0.0
                    if tag_match_count > 0:
                        tag_component = (tag_similarity / tag_match_count) * 0.6
                    
                    # Description component (40% weight)
                    desc_component = 0.0
                    if description_match_count > 0:
                        desc_component = (description_similarity / description_match_count) * 0.4
                    
                    # Combined score
                    semantic_score = tag_component + desc_component
                    
                    # Only include if score is significant
                    if semantic_score > 0.3:
                        scored_semantic_places.append((place, semantic_score))
            else:
                # Text-based fallback matching (for fallback levels 1-3)
                for place in all_places:
                    # Skip if place is already highly ranked in category/tag matches
                    if place in category_tag_places[:int(size * 0.3)]:
                        continue
                    
                    match_score = 0.0
                    tags = place.get("tags", [])
                    description = place.get("description", "")
                    
                    # Tag matching (60% weight)
                    tag_matches = 0
                    for keyword in search_keywords_list:
                        for tag in tags:
                            # Exact or partial match
                            if keyword.lower() in tag.lower() or tag.lower() in keyword.lower():
                                tag_matches += 1
                                break
                    
                    if tag_matches > 0:
                        # Scale by ratio of matched keywords
                        tag_score = min(1.0, tag_matches / len(search_keywords_list))
                        match_score += tag_score * 0.6  # 60% weight
                    
                    # Description matching (40% weight)
                    if description:
                        desc_matches = 0
                        for keyword in search_keywords_list:
                            if keyword.lower() in description.lower():
                                desc_matches += 1
                        
                        if desc_matches > 0:
                            # Scale by ratio of matched keywords
                            desc_score = min(1.0, desc_matches / len(search_keywords_list))
                            match_score += desc_score * 0.4  # 40% weight
                    
                    if match_score > 0.2:  # Lower threshold for text matching
                        scored_semantic_places.append((place, match_score))
            
            # Sort by score
            scored_semantic_places.sort(key=lambda x: x[1], reverse=True)
            semantic_places = [place for place, _ in scored_semantic_places]
            logger.info(f"Found {len(semantic_places)} places via search keywords (fallback level: {fallback_level})")
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        # If semantic search fails, we'll just use the category/tag results
    
    # --- PART 3: COMBINE RESULTS ---
    # Calculate counts for each source
    category_tag_count = min(len(category_tag_places), int(size * 0.6))
    semantic_count = min(len(semantic_places), size - category_tag_count)
    
    # Combine places with no duplicates
    candidate_places = []
    added_ids = set()
    
    # Add category/tag places first (60%)
    for place in category_tag_places[:category_tag_count]:
        place_id = place["_id"]
        if place_id not in added_ids:
            candidate_places.append(place)
            added_ids.add(place_id)
    
    # Add semantic places (40%)
    for place in semantic_places[:semantic_count]:
        place_id = place["_id"]
        if place_id not in added_ids and len(candidate_places) < size:
            candidate_places.append(place)
            added_ids.add(place_id)
    
    # If we don't have enough candidates, add some top-rated places
    if len(candidate_places) < size:
        additional_places = list(
            places_collection.find({"_id": {"$nin": list(added_ids)}})
            .sort([("average_rating", -1)])
            .limit(size - len(candidate_places))
        )
        candidate_places.extend(additional_places)
        logger.info(f"Added {len(additional_places)} additional places based on popularity")
    
    logger.info(f"Returning {len(candidate_places)} total candidate places for user {user_id}")
    return candidate_places




import math
from datetime import datetime, timedelta

def get_collaborative_recommendations(user_id):
    """
    Get place recommendations based on similar users' interactions.
    
    Args:
        user_id: User ID to get recommendations for
        
    Returns:
        List of place IDs recommended through collaborative filtering
    """
    try:
        logger.info(f"Finding collaborative recommendations for user {user_id}")
        user = users_collection.find_one({"_id": user_id})
        if not user:
            logger.warning(f"User {user_id} not found")
            return []

        # Get user preferences
        user_prefs = user.get("preferences", {})
        preferred_categories = user_prefs.get("preferred_categories", [])
        preferred_tags = user_prefs.get("preferred_tags", [])

        # Find similar users using fuzzy matching
        # Use both exact and pattern matches for greater flexibility
        similar_users_query = {
            "_id": {"$ne": user_id},  # Exclude current user
            "$or": []  # Will add conditions below
        }
        
        # Add category conditions if we have categories
        if preferred_categories:
            # Add exact match condition
            similar_users_query["$or"].append(
                {"preferences.preferred_categories": {"$in": preferred_categories}}
            )
            
            # Add partial match conditions for each category
            for category in preferred_categories:
                if category and len(category) > 3:  # Only use meaningful categories
                    # Find users with categories that contain this category as a substring
                    similar_users_query["$or"].append(
                        {"preferences.preferred_categories": {"$regex": category, "$options": "i"}}
                    )
        
        # Add tag conditions if we have tags
        if preferred_tags:
            # Add exact match condition
            similar_users_query["$or"].append(
                {"preferences.preferred_tags": {"$in": preferred_tags}}
            )
            
            # Add partial match conditions for each tag
            for tag in preferred_tags:
                if tag and len(tag) > 3:  # Only use meaningful tags
                    # Find users with tags that contain this tag as a substring
                    similar_users_query["$or"].append(
                        {"preferences.preferred_tags": {"$regex": tag, "$options": "i"}}
                    )
        
        # If we have no query conditions, use a reasonable default
        if not similar_users_query["$or"]:
            similar_users = list(users_collection.find({"_id": {"$ne": user_id}}).limit(50))
        else:
            similar_users = list(users_collection.find(similar_users_query).limit(100))  # Increased limit for broader matches

        logger.info(f"Found {len(similar_users)} similar users for user {user_id}")

        # Define weights for different interaction types
        action_weights = {
            "like": 5,
            "save": 4,
            "share": 3,
            "comment": 3,
            "view": 2,
            "click": 1,
            "dislike": -5
        }

        # Get current time consistently as timezone-naive
        current_date = datetime.now().replace(tzinfo=None).date()

        # Time decay factor for older interactions
        def apply_time_decay(weight, interaction_time):
            # Parse the timestamp string if needed
            if isinstance(interaction_time, str):
                try:
                    # Try to parse string timestamp and remove timezone
                    timestamp = interaction_time.split("T")[0]  # Take just the date part
                    interaction_time = datetime.strptime(timestamp, "%Y-%m-%d").date()
                except Exception as e:
                    # If parsing fails, use current date
                    interaction_time = current_date
            # If it's already a datetime, convert to date
            elif hasattr(interaction_time, 'date'):
                try:
                    interaction_time = interaction_time.replace(tzinfo=None).date()
                except:
                    interaction_time = current_date
            else:
                # Fallback to current date
                interaction_time = current_date

            # Calculate days between dates
            days_ago = max(0, (current_date - interaction_time).days)
            decay = math.exp(-days_ago / 30)  # Exponential decay over 30 days
            return weight * decay

        # Calculate similarity scores for each user
        user_similarities = {}
        for similar_user in similar_users:
            similarity = calculate_similarity_score(user, similar_user)
            user_similarities[similar_user["_id"]] = similarity

        # Track recommended places with scores
        place_scores = {}

        # Get existing interactions for user
        user_interactions = {}
        for i in interactions_collection.find({"user_id": user_id}):
            if "place_id" in i and "interaction_type" in i:
                user_interactions[i["place_id"]] = i["interaction_type"]

        # Process interactions from similar users
        for similar_user in similar_users:
            interactions = list(interactions_collection.find({"user_id": similar_user["_id"]}).limit(100))
            
            # Get similarity score for this user
            similarity = user_similarities.get(similar_user["_id"], 0.5)  # Default 0.5 if missing

            for interaction in interactions:
                # Skip if place_id or interaction_type is missing
                if "place_id" not in interaction or "interaction_type" not in interaction:
                    continue

                place_id = interaction["place_id"]
                action = interaction["interaction_type"]

                # Get timestamp with fallback to current date
                timestamp = interaction.get("timestamp", current_date)

                # Skip if user already dislikes this place
                if place_id in user_interactions and user_interactions[place_id] == "dislike":
                    continue

                weight = action_weights.get(action, 1)  # Default weight of 1 for unknown actions
                
                # Apply time decay
                time_decayed_score = apply_time_decay(weight, timestamp)
                
                # Apply user similarity as a multiplier
                final_score = time_decayed_score * similarity

                # Only add positively scored places
                if final_score > 0:
                    if place_id not in place_scores:
                        place_scores[place_id] = 0
                    place_scores[place_id] += final_score

        # Sort places by score
        sorted_places = sorted(place_scores.items(), key=lambda x: x[1], reverse=True)
        recommended_place_ids = [place_id for place_id, _ in sorted_places]

        logger.info(f"Found {len(recommended_place_ids)} collaborative recommendations for user {user_id}")
        return recommended_place_ids
    except Exception as e:
        logger.error(f"Error in collaborative filtering: {str(e)}")
        return []

def calculate_similarity_score(user1, user2):
    """
    Calculate similarity between two users based on their preferences.
    Returns a score between 0 and 1.
    """
    try:
        # Get preferences
        prefs1 = user1.get("preferences", {})
        prefs2 = user2.get("preferences", {})
        
        if not prefs1 or not prefs2:
            return 0.3  # Default modest similarity when preferences missing
        
        similarity_score = 0.0
        factors_count = 0
        
        # 1. Category similarity (weighted 40%)
        cats1 = set(prefs1.get("preferred_categories", []))
        cats2 = set(prefs2.get("preferred_categories", []))
        
        if cats1 and cats2:
            # Jaccard similarity for categories
            category_jaccard = len(cats1.intersection(cats2)) / max(len(cats1.union(cats2)), 1)
            similarity_score += category_jaccard * 0.4
            factors_count += 1
        
        # 2. Tag similarity (weighted 40%)
        tags1 = set(prefs1.get("preferred_tags", []))
        tags2 = set(prefs2.get("preferred_tags", []))
        
        if tags1 and tags2:
            # Jaccard similarity for tags
            tag_jaccard = len(tags1.intersection(tags2)) / max(len(tags1.union(tags2)), 1)
            similarity_score += tag_jaccard * 0.4
            factors_count += 1
        
        # 3. Activity level similarity (weighted 20%)
        # This is a proxy for user engagement patterns
        activity1 = prefs1.get("activity_level", "medium")
        activity2 = prefs2.get("activity_level", "medium")
        
        activity_score = 0
        if activity1 == activity2:
            activity_score = 1.0
        elif (activity1 in ["high", "medium"] and activity2 in ["high", "medium"]) or \
             (activity1 in ["medium", "low"] and activity2 in ["medium", "low"]):
            # Adjacent activity levels
            activity_score = 0.5
            
        similarity_score += activity_score * 0.2
        factors_count += 1
        
        # Normalize if we have factors
        final_score = similarity_score / factors_count if factors_count > 0 else 0.3
        
        # Boost score slightly to encourage more recommendations
        boosted_score = min(1.0, final_score * 1.2)
        
        return boosted_score
        
    except Exception as e:
        logger.error(f"Error calculating user similarity: {str(e)}")
        return 0.3  # Default modest similarity on error
def get_discovery_places(user_id, limit=10):
    """Get places outside the user's normal patterns for discovery"""
    try:
        user_prefs = get_user_travel_preferences(user_id)
        if not user_prefs:
            return []

        preferred_categories = user_prefs.get("preferred_categories", [])
        preferred_tags = user_prefs.get("preferred_tags", [])

        # Find places in different categories but highly rated
        discovery_query = {
            "$and": [
                {"category": {"$nin": preferred_categories}},
                {"average_rating": {"$gte": 4.0}}  # Only high-rated places
            ]
        }

        # Get discovery places and sort by rating
        discovery_places = list(places_collection.find(discovery_query).sort("average_rating", -1).limit(limit * 2))

        # If we don't have enough, try a broader search
        if len(discovery_places) < limit:
            fallback_places = list(places_collection.find({
                "category": {"$nin": preferred_categories}
            }).sort("average_rating", -1).limit(limit * 2))

            # Add any new places not already in discovery_places
            existing_ids = set(p["_id"] for p in discovery_places)
            for place in fallback_places:
                if place["_id"] not in existing_ids:
                    discovery_places.append(place)
                    if len(discovery_places) >= limit * 2:
                        break

        # Randomize the order for more variety in recommendations
        if discovery_places:
            random.shuffle(discovery_places)

        logger.info(f"Found {len(discovery_places[:limit])} discovery places for user {user_id}")
        return discovery_places[:limit]
    except Exception as e:
        logger.error(f"Error getting discovery places: {str(e)}")
        return []
def calculate_personalization_score(place, user_id, user_prefs):
    """
    Calculate personalization score for a place based on user preferences.
    
    Scoring components:
    - Category match (40%): Direct match (1.0) or partial/substring match (0.7)
    - Tag match (30%): Proportional to number of matching tags
    - Rating factor (20%): Normalized place rating (0-5 scale)
    - User interaction history (10%): Based on previous positive interactions or dislikes
    
    The final score is weighted: (category*0.4 + tags*0.3 + rating*0.2 + interactions*0.1)
    
    Args:
        place: Place document
        user_id: User ID
        user_prefs: User preferences dictionary
        
    Returns:
        Personalization score between 0 and 1
    """
    try:
        # 1. Category matching (40% of score)
        category_score = 0
        place_category = place.get("category", "").lower()
        preferred_categories = [cat.lower() for cat in user_prefs.get("preferred_categories", [])]
        
        if preferred_categories:
            # Direct category match
            if place_category in preferred_categories:
                category_score = 1.0
            else:
                # Check for partial matches (e.g., "beach resort" contains "beach")
                for category in preferred_categories:
                    if category in place_category or place_category in category:
                        category_score = 0.7
                        break
        else:
            # No preferred categories, neutral score
            category_score = 0.5
            
        # 2. Tag matching (30% of score)
        tag_score = 0
        place_tags = [tag.lower() for tag in place.get("tags", [])]
        preferred_tags = [tag.lower() for tag in user_prefs.get("preferred_tags", [])]
        
        if preferred_tags and place_tags:
            matching_tags = set(place_tags).intersection(set(preferred_tags))
            tag_score = len(matching_tags) / max(len(preferred_tags), 1)
        else:
            # No tags to compare, neutral score
            tag_score = 0.5
            
        # 3. Rating factor (20% of score)
        rating_score = min(place.get("rating", 0) / 5.0, 1.0)  # Normalize to 0-1
            
        # 4. User interaction history (10% of score)
        interaction_score = 0.5  # Default neutral score
        
        # Look up past interactions
        past_interactions = interactions_collection.find_one({
            "user_id": user_id,
            "place_id": place["_id"]
        })
        
        if past_interactions:
            # Positive interactions increase score
            if past_interactions.get("liked", False):
                interaction_score = 0.9
            elif past_interactions.get("saved", False):
                interaction_score = 0.8
            elif past_interactions.get("viewed", 0) > 3:
                interaction_score = 0.7
            # Negative interactions decrease score
            elif past_interactions.get("disliked", False):
                interaction_score = 0.1
        
        # Calculate final weighted score
        final_score = (
            (category_score * 0.4) +
            (tag_score * 0.3) +
            (rating_score * 0.2) +
            (interaction_score * 0.1)
        )
        
        return final_score
        
    except Exception as e:
        logger.error(f"Error calculating personalization score: {e}")
        return 0.5  # Return neutral score on error

def rank_places(candidate_places, user_id):
    """
    Rank places based on user engagement and popularity metrics.
    
    Args:
        candidate_places: List of place documents to rank
        user_id: User ID for personalization
        
    Returns:
        List of places sorted by relevance score
    """
    # Helper function to extract numeric value from MongoDB document fields
    def extract_number(value):
        if isinstance(value, dict):
            # Handle MongoDB numeric types
            if "$numberDouble" in value:
                return float(value["$numberDouble"])
            if "$numberInt" in value:
                return int(value["$numberInt"])
            if "$numberLong" in value:
                return int(value["$numberLong"])
        return value or 0  # Return the value itself if not a dict, or 0 if None/falsy

    try:
        if not candidate_places:
            return []

        scaler = MinMaxScaler()

        for place in candidate_places:
            interactions_count = interactions_collection.count_documents(
                {"user_id": user_id, "place_id": place["_id"]}
            )

            # Extract numeric values correctly
            avg_rating = extract_number(place.get("average_rating", 0))
            likes = extract_number(place.get("likes", 0))

            # Calculate raw score using weights:
            # - Rating: 40%
            # - Likes: 30%
            # - User interactions: 30%
            place["score"] = (
                0.4 * avg_rating +  # Rating weight
                0.3 * likes / 10000 +  # Normalize likes (assume 10K is max)
                0.3 * interactions_count  # User interaction weight
            )

        # Normalize scores using MinMaxScaler
        scores = [[p["score"]] for p in candidate_places]
        if scores:
            try:
                scaled_scores = scaler.fit_transform(scores)
                for i, place in enumerate(candidate_places):
                    place["final_score"] = float(scaled_scores[i][0])  # Convert numpy type to float
            except Exception as e:
                logger.error(f"Error scaling scores: {e}")
                # Fallback to unscaled scores
                for place in candidate_places:
                    place["final_score"] = place["score"]
        else:
            for place in candidate_places:
                place["final_score"] = 0  # Default score if no data available

        # Use final_score for sorting, not the objects themselves
        return sorted(candidate_places, key=lambda x: x.get("final_score", 0), reverse=True)
    except Exception as e:
        logger.error(f"Error ranking places: {e}")
        # Use average_rating for fallback sorting, but extract numeric value first
        return sorted(candidate_places,
                     key=lambda x: extract_number(x.get("average_rating", 0)),
                     reverse=True)  # Fallback sorting

def get_previously_shown_places(user_id):
    """
    Get a list of all place IDs previously shown to the user.
    
    Args:
        user_id: User ID
        
    Returns:
        List of place IDs
    """
    try:
        user_shown = shown_places_collection.find_one({"user_id": user_id})
        return user_shown.get("place_ids", []) if user_shown else []
    except Exception as e:
        logger.error(f"Error getting previously shown places: {e}")
        return []

def get_last_shown_places(user_id):
    """
    Get only the places shown in the most recent request.
    
    Args:
        user_id: User ID
        
    Returns:
        List of place IDs from last request
    """
    try:
        user_shown = shown_places_collection.find_one({"user_id": user_id})
        return user_shown.get("last_shown_place_ids", []) if user_shown else []
    except Exception as e:
        logger.error(f"Error getting last shown places: {e}")
        return []

def reset_user_shown_places(user_id):
    """
    Reset the tracking of places shown to a user.
    
    Args:
        user_id: User ID
        
    Returns:
        Boolean indicating success
    """
    try:
        result = shown_places_collection.delete_one({"user_id": user_id})
        deleted = result.deleted_count > 0
        logger.info(f"Reset shown places for user {user_id}, success: {deleted}")
        return deleted
    except Exception as e:
        logger.error(f"Error resetting shown places: {e}")
        return False

def update_shown_places(user_id, new_place_ids, max_places=100):
    """
    Update the list of shown places for a user.
    Keeps track of places in chronological order, with most recent at the END of the list.
    
    Args:
        user_id: User ID
        new_place_ids: List of new place IDs to add to shown list
        max_places: Maximum number of places to keep in history
    """
    try:
        # Get current shown places
        shown_doc = shown_places_collection.find_one({"user_id": user_id})
        
        if shown_doc:
            # Get existing place IDs
            existing_ids = shown_doc.get("place_ids", [])
            
            # Remove new IDs if they already exist (to avoid duplicates)
            existing_ids = [pid for pid in existing_ids if pid not in new_place_ids]
            
            # Add new IDs at the END of the list (most recent)
            updated_ids = existing_ids + new_place_ids
            
            # Keep only the most recent max_places
            if len(updated_ids) > max_places:
                updated_ids = updated_ids[-max_places:]
            
            # Update document with new list and last shown
            shown_places_collection.update_one(
                {"user_id": user_id},
                {"$set": {
                    "place_ids": updated_ids, 
                    "last_shown_place_ids": new_place_ids,
                    "last_updated": datetime.now()
                }}
            )
        else:
            # Create new document
            shown_places_collection.insert_one({
                "user_id": user_id,
                "place_ids": new_place_ids,
                "last_shown_place_ids": new_place_ids,
                "last_updated": datetime.now()
            })
            
        logger.info(f"Updated shown places for user {user_id}, added {len(new_place_ids)} places")
    except Exception as e:
        logger.error(f"Error updating shown places: {str(e)}")
def get_unshown_places(user_id, limit=10):
    """
    Get places that haven't been shown to the user yet.
    
    Args:
        user_id: User ID
        limit: Maximum number of places to return
        
    Returns:
        List of place documents
    """
    try:
        # Get previously shown places
        shown_place_ids = get_previously_shown_places(user_id)

        # Get all places that haven't been shown to this user
        if shown_place_ids:
            unshown_places = list(places_collection.find({"_id": {"$nin": shown_place_ids}}).limit(limit))
        else:
            # If no shown places, get any places
            unshown_places = list(places_collection.find().limit(limit))

        return unshown_places
    except Exception as e:
        logger.error(f"Error getting unshown places: {e}")
        return []

def refresh_shown_places(user_id, shown_place_ids, limit=10):
    """
    Re-rank and refresh previously shown places based on recent activity.
    
    Args:
        user_id: User ID
        shown_place_ids: List of place IDs to refresh
        limit: Maximum number of places to return
        
    Returns:
        List of refreshed place documents
    """
    try:
        if not shown_place_ids:
            return []

        # Get the place documents
        shown_places = list(places_collection.find({"_id": {"$in": shown_place_ids}}))

        if not shown_places:
            return []

        # Get recent interaction data for all users
        recent_date = datetime.now() - timedelta(days=7)

        # Count recent interactions for each place
        place_interaction_counts = {}
        for place_id in shown_place_ids:
            count = interactions_collection.count_documents({
                "place_id": place_id,
                "timestamp": {"$gte": recent_date},
                "interaction_type": {"$in": ["like", "save", "share", "view"]}
            })
            place_interaction_counts[place_id] = count

        # Add recency score to places
        for place in shown_places:
            place["recency_score"] = place_interaction_counts.get(place["_id"], 0)

        # Sort by recency score and add some randomness for variety
        refreshed_places = sorted(
            shown_places, 
            key=lambda x: x.get("recency_score", 0) + random.random(), 
            reverse=True
        )

        return refreshed_places[:limit]
    except Exception as e:
        logger.error(f"Error refreshing shown places: {e}")
        return []
# --- PART 4: RECOMMENDATION GENERATION AND CACHING ---
def generate_final_recommendations(user_id, num_recommendations=10, previously_shown_ids=None):
    """
    Generate final personalized recommendations for a user.
    Enhanced with collaborative filtering (40% of recommendations) and explicit tracking of shown places.
    Implements advanced fallback mechanisms with randomized mixing when running out of new places.
    
    Args:
        user_id: User ID
        num_recommendations: Number of recommendations to generate
        previously_shown_ids: List of previously shown place IDs to exclude
        
    Returns:
        List of personalized recommendations
    """
    try:
        logger.info(f"Generating final recommendations for user {user_id}, need {num_recommendations}")
        
        # Get user preferences
        user_prefs = get_user_travel_preferences(user_id)
        
        # Use provided previously_shown_ids or fetch them if not provided
        if previously_shown_ids is None:
            previously_shown = shown_places_collection.find_one({"user_id": user_id})
            previously_shown_ids = previously_shown.get("place_ids", []) if previously_shown else []
        
        # Initialize recommendations
        recommendations = []
        
        # PART 1: Calculate recommendation distribution (40% collaborative, 60% content-based)
        collab_count = int(num_recommendations * 0.4)  # 40% for collaborative
        content_count = num_recommendations - collab_count  # 60% for content-based
        
        # PART 2: Get collaborative recommendations first (40%)
        if collab_count > 0:
            collab_places_ids = get_collaborative_recommendations(user_id)
            
            # Log the total number of collaborative recommendations found
            logger.info(f"Found {len(collab_places_ids)} collaborative recommendations for user {user_id}")
            
            # Filter out previously shown places
            collab_places_ids = [pid for pid in collab_places_ids if pid not in previously_shown_ids]
            
            # Get place details for remaining collaborative recommendations
            if collab_places_ids:
                # Get all collaborative place details
                all_collab_places = list(places_collection.find({"_id": {"$in": collab_places_ids}}))
                
                # Sort collaborative recommendations by rating for consistency
                all_collab_places.sort(key=lambda x: x.get("average_rating", 0), reverse=True)
                
                # Use top collaborative places up to our limit
                collab_to_add = all_collab_places[:collab_count]
                
                # Add collaborative places
                for place in collab_to_add:
                    place["source"] = "collaborative"
                    recommendations.append(place)
                
                logger.info(f"Added {len(collab_to_add)} collaborative filtering recommendations")
            else:
                logger.info("No new collaborative recommendations available")
        
        # PART 3: Get content-based recommendations (60% or more if collaborative failed)
        remaining_content_count = num_recommendations - len(recommendations)
        
        if remaining_content_count > 0:
            # Get candidate places
            candidate_places = get_candidate_places(user_prefs, user_id, size=remaining_content_count * 5)
            
            # Filter out previously shown places and already added recommendations
            filtered_candidates = []
            for place in candidate_places:
                if place["_id"] not in previously_shown_ids and place["_id"] not in [r["_id"] for r in recommendations]:
                    filtered_candidates.append(place)
            
            # Apply personalization factors to remaining candidates
            ranked_places = []
            
            for place in filtered_candidates:
                # Calculate personalization score
                score = calculate_personalization_score(place, user_id, user_prefs)
                ranked_places.append((place, score))
            
            # Sort places by personalization score
            ranked_places.sort(key=lambda x: x[1], reverse=True)
            
            # Add top content-based places
            added_content_places = 0
            for place, _ in ranked_places:
                if len(recommendations) < num_recommendations:
                    place["source"] = "content_based"
                    recommendations.append(place)
                    added_content_places += 1
                else:
                    break
            
            logger.info(f"Added {added_content_places} content-based places")
        
        # PART 4: FALLBACK OPTIONS - Prepare all potential fallback sources
        # If we still need more recommendations, prepare fallback options
        if len(recommendations) < num_recommendations:
            remaining_needed = num_recommendations - len(recommendations)
            logger.info(f"Need {remaining_needed} more recommendations, preparing fallback options")
            
            # Keep track of IDs we've already included
            current_rec_ids = [r["_id"] for r in recommendations]
            
            # FALLBACK SOURCE 1: Discovery places
            discovery_places = []
            try:
                raw_discovery = get_discovery_places(user_id, limit=remaining_needed * 2)
                for place in raw_discovery:
                    if place["_id"] not in previously_shown_ids and place["_id"] not in current_rec_ids:
                        place["source"] = "discovery"
                        discovery_places.append(place)
            except Exception as e:
                logger.error(f"Error getting discovery places: {str(e)}")
            
            # FALLBACK SOURCE 2: Trending places
            trending_places = []
            try:
                # Get recent interactions to find trending places
                recent_date = datetime.now() - timedelta(days=7)
                trending_interactions = list(interactions_collection.aggregate([
                    {"$match": {"timestamp": {"$gte": recent_date}}},
                    {"$group": {"_id": "$place_id", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                    {"$limit": remaining_needed * 2}
                ]))
                
                trending_place_ids = [item["_id"] for item in trending_interactions]
                trending_place_ids = [pid for pid in trending_place_ids 
                                    if pid not in previously_shown_ids and 
                                       pid not in current_rec_ids]
                
                if trending_place_ids:
                    raw_trending = list(places_collection.find({"_id": {"$in": trending_place_ids}}))
                    for place in raw_trending:
                        place["source"] = "trending"
                        trending_places.append(place)
            except Exception as e:
                logger.error(f"Error getting trending places: {str(e)}")
            
            # FALLBACK SOURCE 3: Old previously shown places (user likely forgot)
            old_shown_places = []
            try:
                if previously_shown_ids and len(previously_shown_ids) > 20:
                    # Get places from beginning of history (oldest ones user likely forgot)
                    # Use the first 30% of the history
                    oldest_count = min(int(len(previously_shown_ids) * 0.3), remaining_needed * 2)
                    oldest_ids = previously_shown_ids[:oldest_count]
                    
                    # Filter out IDs already in recommendations
                    oldest_ids = [pid for pid in oldest_ids if pid not in current_rec_ids]
                    
                    if oldest_ids:
                        raw_old_places = list(places_collection.find({"_id": {"$in": oldest_ids}}))
                        for place in raw_old_places:
                            place["source"] = "rediscovery"
                            old_shown_places.append(place)
            except Exception as e:
                logger.error(f"Error getting old shown places: {str(e)}")
            
            # FALLBACK SOURCE 4: Any top-rated places as last resort
            top_rated_places = []
            try:
                # Get top-rated places not already included
                raw_top_rated = list(places_collection.find(
                    {"_id": {"$nin": current_rec_ids + previously_shown_ids}}
                ).sort("average_rating", -1).limit(remaining_needed))
                
                for place in raw_top_rated:
                    place["source"] = "top_rated"
                    top_rated_places.append(place)
            except Exception as e:
                logger.error(f"Error getting top-rated places: {str(e)}")
            
            # PART 5: COMBINE FALLBACK SOURCES - Mix randomly to avoid patterns
            # Combine all fallback options
            all_fallbacks = discovery_places + trending_places + old_shown_places + top_rated_places
            
            # Shuffle to randomize selection (avoid sequential patterns)
            random.shuffle(all_fallbacks)
            
            # Remove any duplicates by ID
            seen_ids = set(current_rec_ids)
            unique_fallbacks = []
            for place in all_fallbacks:
                if place["_id"] not in seen_ids:
                    seen_ids.add(place["_id"])
                    unique_fallbacks.append(place)
            
            # Add fallbacks until we reach target count
            added_fallback_count = 0
            for place in unique_fallbacks:
                if len(recommendations) < num_recommendations:
                    recommendations.append(place)
                    added_fallback_count += 1
                else:
                    break
            
            logger.info(f"Added {added_fallback_count} mixed fallback recommendations")
        
        # PART 6: ABSOLUTE LAST RESORT - Reuse any places if we still need more
        if len(recommendations) < num_recommendations:
            remaining_needed = num_recommendations - len(recommendations)
            logger.info(f"Still need {remaining_needed} more places, using any available places")
            
            # Get any places excluding what we've already added
            current_rec_ids = [r["_id"] for r in recommendations]
            last_resort_places = list(places_collection.find(
                {"_id": {"$nin": current_rec_ids}}
            ).sort("average_rating", -1).limit(remaining_needed))
            
            for place in last_resort_places:
                place["source"] = "last_resort"
                recommendations.append(place)
            
            logger.info(f"Added {len(last_resort_places)} last resort places")
        
        logger.info(f"Final recommendation count: {len(recommendations)}/{num_recommendations}")
        return recommendations[:num_recommendations]
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        # Return some fallback recommendations in case of error
        fallback_places = list(places_collection.find().sort("average_rating", -1).limit(num_recommendations))
        for place in fallback_places:
            place["source"] = "error_fallback"
        return fallback_places
def get_recommendations_with_caching(user_id, force_refresh=False, num_new_recommendations=10, max_total=30):
    """
    Get recommendations for a user with caching and progressive pagination.
    First request: Return 10 new places
    Second request: Return 10 new places + previous 10 = 20 total
    Third request and beyond: Return 10 new places + 20 most recent shown places = 30 total
    
    Args:
        user_id: User ID
        force_refresh: Whether to force generation of new recommendations
        num_new_recommendations: Number of new recommendations to return (default 10)
        max_total: Maximum total recommendations to return (default 30)
        
    Returns:
        Dict with new recommendations and previously shown recommendations
    """
    try:
        # Get previously shown places
        previously_shown = shown_places_collection.find_one({"user_id": user_id})
        previously_shown_ids = previously_shown.get("place_ids", []) if previously_shown else []
        
        # Determine how many previously shown places to return based on pagination
        total_shown_count = len(previously_shown_ids)
        
        # Progressive pagination logic:
        # - First request: Return 10 new places only (0 previous)
        # - Second request: Return 10 new + 10 previous = 20 total
        # - Third+ request: Return 10 new + 20 previous = 30 total
        if total_shown_count == 0:
            # First request - no previous places
            history_count = 0
        elif total_shown_count <= 10:
            # Second request - include up to 10 previous places
            history_count = total_shown_count
        else:
            # Third+ request - include up to 20 previous places
            history_count = min(20, total_shown_count)
        
        logger.info(f"User {user_id} pagination: {total_shown_count} total shown, returning {history_count} history items")
        
        # Get new recommendations (either from cache or generated)
        new_recommendations = []
        
        if force_refresh:
            logger.info(f"Force refresh requested for user {user_id}")
            new_recommendations = generate_final_recommendations(user_id, num_new_recommendations, previously_shown_ids)
        else:
            # Look for cached recommendations
            cached_entries = get_user_cached_recommendations(user_id)
            
            if not cached_entries:
                logger.info(f"No cached recommendations found for user {user_id}")
                new_recommendations = generate_final_recommendations(user_id, num_new_recommendations, previously_shown_ids)
            else:
                # Get the first entry with lowest sequence number
                cached_entry = cached_entries[0]
                
                # Remove the used entry from cache
                recommendations_cache_collection.delete_one({"_id": cached_entry["_id"]})
                
                logger.info(f"Using cached recommendations for user {user_id} (sequence {cached_entry['sequence']})")
                
                # If this was the last entry, scheduling will be handled by the endpoint
                if len(cached_entries) <= 2:
                    logger.info(f"Only {len(cached_entries)} cached entries left for user {user_id}, scheduling more")
                
                # Filter cached recommendations to ensure they haven't been shown before
                filtered_cache = []
                for place in cached_entry["recommendations"]:
                    if place["_id"] not in previously_shown_ids:
                        filtered_cache.append(place)
                
                # If we filtered too many, generate new ones
                if len(filtered_cache) >= num_new_recommendations:
                    new_recommendations = filtered_cache[:num_new_recommendations]
                else:
                    # Not enough from cache after filtering, generate new ones
                    logger.info(f"Only {len(filtered_cache)} new places in cache after filtering, generating more")
                    additional_needed = num_new_recommendations - len(filtered_cache)
                    
                    # Add what we have from cache
                    new_recommendations = filtered_cache.copy()
                    
                    # Generate more to supplement
                    existing_ids = [p["_id"] for p in new_recommendations]
                    additional_recs = generate_final_recommendations(
                        user_id, 
                        additional_needed, 
                        previously_shown_ids + existing_ids
                    )
                    
                    # Add the additional recommendations
                    new_recommendations.extend(additional_recs)
        
        # Get the IDs of new recommendations
        new_place_ids = [p["_id"] for p in new_recommendations]
        
        # Get previously shown places for history display (most recent N, excluding new places)
        previously_shown_places = []
        if history_count > 0:
            # Filter out places we're showing as new
            shown_ids_to_fetch = [pid for pid in previously_shown_ids if pid not in new_place_ids]
            
            # Get the most recent ones based on history_count (from end of list)
            if shown_ids_to_fetch:
                # Get the most recent items (last N items in the list)
                shown_ids_to_fetch = shown_ids_to_fetch[-history_count:]
                
                # Fetch the actual place data
                if shown_ids_to_fetch:
                    previously_shown_places = list(places_collection.find({"_id": {"$in": shown_ids_to_fetch}}))
                    
                    # Add source information
                    for place in previously_shown_places:
                        place["source"] = "history"
        
        # Track these new recommendations as shown
        update_shown_places(user_id, new_place_ids, max_places=100)
        
        return {
            "new_recommendations": new_recommendations,
            "previously_shown": previously_shown_places
        }
            
    except Exception as e:
        logger.error(f"Error getting recommendations with caching: {str(e)}")
        # Fallback to generating without cache
        new_recommendations = generate_final_recommendations(user_id, num_new_recommendations, [])
        return {
            "new_recommendations": new_recommendations,
            "previously_shown": []
        }
async def background_cache_recommendations(user_id, num_entries=6):
    """
    Background task to pre-generate multiple recommendation entries for caching.
    Generates num_entries separate entries, each with a different set of recommendations.
    
    Args:
        user_id: User ID to generate recommendations for
        num_entries: Number of cache entries to generate
    """
    # Check if generation is already in progress
    cache_lock_key = f"cache_lock_{user_id}"
    
    try:
        # Try to acquire the lock
        lock_result = cache_locks_collection.update_one(
            {"_id": cache_lock_key, "user_id": user_id, "locked": {"$ne": True}},
            {"$set": {
                "user_id": user_id,
                "locked": True,
                "timestamp": datetime.now()
            }},
            upsert=True
        )
        
        # Check if lock was acquired
        if lock_result.modified_count == 0 and lock_result.upserted_id is None:
            # Check if lock is stale (older than 5 minutes)
            lock = cache_locks_collection.find_one({"_id": cache_lock_key})
            if lock:
                lock_time = lock.get("timestamp", datetime.min)
                if isinstance(lock_time, str):
                    try:
                        lock_time = datetime.fromisoformat(lock_time.replace('Z', '+00:00'))
                    except Exception:
                        lock_time = datetime.min
                        
                if (datetime.now() - lock_time).total_seconds() < 300:  # 5 minutes
                    logger.info(f"Cache generation already in progress for user {user_id}, skipping")
                    return
                
                # Lock is stale, force delete it
                cache_locks_collection.delete_one({"_id": cache_lock_key})
                logger.info(f"Removed stale lock for user {user_id}")
                
                # Try to acquire the lock again
                cache_locks_collection.insert_one({
                    "_id": cache_lock_key,
                    "user_id": user_id,
                    "locked": True,
                    "timestamp": datetime.now()
                })
            else:
                logger.info(f"Cache generation already in progress for user {user_id}, skipping")
                return
            
        # Get existing entries to avoid duplicates
        try:
            existing_entries = get_user_cached_recommendations(user_id)
            existing_sequences = {entry["sequence"] for entry in existing_entries}
            
            max_sequence = max(existing_sequences) if existing_sequences else 0
            next_sequence = max_sequence + 1
        except Exception as e:
            logger.error(f"Error getting existing cache entries: {e}")
            next_sequence = 0
            
        # Generate recommendations in sequence
        logger.info(f"Generating {num_entries} cache entries for user {user_id}")
        
        for i in range(num_entries):
            try:
                # Generate with slightly different weights for variety
                randomization_seed = next_sequence + i + int(datetime.now().timestamp())
                random.seed(randomization_seed)
                
                # Wait a small amount of time between generations for variety
                await asyncio.sleep(0.5)
                
                # Generate fresh recommendations with randomized weights
                collab_weight = 0.4 + (random.random() * 0.1 - 0.05)  # 35-45%
                recommendations = generate_final_recommendations(user_id, 10)
                
                # Store in cache with incrementing sequence
                sequence = next_sequence + i
                
                # Store with generation parameters for debugging
                recommendations_cache_collection.insert_one({
                    "user_id": user_id,
                    "sequence": sequence,
                    "recommendations": recommendations,
                    "timestamp": datetime.now(),
                    "generation_params": {
                        "collab_weight": collab_weight,
                        "randomization_seed": randomization_seed
                    }
                })
                
                logger.info(f"Generated cache entry {i+1}/{num_entries} for user {user_id} (sequence {sequence})")
                
            except Exception as entry_error:
                logger.error(f"Error generating cache entry {i+1}/{num_entries}: {entry_error}")
                
    except Exception as e:
        logger.error(f"Error in background cache task: {e}")
        
    finally:
        # Always release the lock
        try:
            cache_locks_collection.delete_one({"_id": cache_lock_key})
            logger.info(f"Released cache lock for user {user_id}")
        except Exception as lock_error:
            logger.error(f"Error releasing cache lock: {lock_error}")
# --- PART 5: ROADMAP GENERATION ---

def get_seasonal_activities(month=None):
    """
    Get seasonal activities based on the current month or specified month.
    
    Args:
        month: Month name (e.g. "January"), defaults to current month
        
    Returns:
        Dictionary mapping seasons to weights
    """
    if not month:
        # Default to current month
        month = datetime.now().strftime("%B")
    
    # Map months to seasons with weights
    season_weights = {
        "winter": 0,
        "spring": 0,
        "summer": 0,
        "fall": 0
    }
    
    # Northern hemisphere seasons
    if month in ["December", "January", "February"]:
        season_weights["winter"] = 1.0
        season_weights["fall"] = 0.3
    elif month in ["March", "April", "May"]:
        season_weights["spring"] = 1.0
        season_weights["winter"] = 0.3
    elif month in ["June", "July", "August"]:
        season_weights["summer"] = 1.0
        season_weights["spring"] = 0.3
    elif month in ["September", "October", "November"]:
        season_weights["fall"] = 1.0
        season_weights["summer"] = 0.3
    
    return season_weights

def get_budget_mappings():
    """
    Get budget level mappings.
    
    Returns:
        Dictionary of budget levels and their numerical mappings
    """
    return {
        "budget": 1,
        "low": 1,
        "economy": 1,
        "mid-range": 2,
        "medium": 2,
        "moderate": 2,
        "high-end": 3,
        "high": 3,
        "luxury": 4,
        "premium": 4,
        "exclusive": 5
    }

def map_budget_level(budget_text):
    """
    Map a text budget description to a numerical level (1-5).
    
    Args:
        budget_text: Text description of budget
        
    Returns:
        Numerical budget level from 1 (lowest) to 5 (highest)
    """
    if not budget_text:
        return 3  # Default to medium
    
    budget_text = budget_text.lower().strip()
    budget_mappings = get_budget_mappings()
    
    # Direct match
    if budget_text in budget_mappings:
        return budget_mappings[budget_text]
    
    # Substring match
    for key, value in budget_mappings.items():
        if key in budget_text:
            return value
    
    # Default to medium if no match
    return 3

def calculate_budget_compatibility(place_budget_level, user_budget_level):
    """
    Calculate compatibility score between place budget and user budget.
    
    Args:
        place_budget_level: Numerical budget level of place (1-5)
        user_budget_level: Numerical budget level of user (1-5)
        
    Returns:
        Compatibility score between 0 and 1
    """
    # Standardize budget levels
    if isinstance(place_budget_level, str):
        place_budget_level = map_budget_level(place_budget_level)
    if isinstance(user_budget_level, str):
        user_budget_level = map_budget_level(user_budget_level)
    
    # Calculate the absolute difference (0-4)
    diff = abs(place_budget_level - user_budget_level)
    
    # Convert to 0-1 score (1 is perfect match, 0 is worst match)
    # Maximum difference is 4, so we do 1 - (diff/4)
    score = 1 - (diff / 4)
    
    return score

def check_accessibility_compatibility(place, accessibility_needs):
    """
    Check if a place is compatible with the user's accessibility needs.
    
    Args:
        place: Place document
        accessibility_needs: List of accessibility requirements
        
    Returns:
        Boolean indicating whether the place meets all accessibility requirements
    """
    if not accessibility_needs:
        return True  # No requirements, so all places are compatible
    
    # Get place accessibility features
    place_features = place.get("accessibility_features", [])
    
    # Check if all required features are present
    for need in accessibility_needs:
        if need not in place_features:
            return False
    
    return True

def generate_hybrid_roadmap(user_id):
    """
    Generate a travel roadmap for a user using a hybrid two-stage filtering approach.
    First applies critical filters, then soft constraints with weighted scoring.
    Always returns exactly 10 places, using nearby places and trending places as fallbacks.
    
    Args:
        user_id: User ID
        
    Returns:
        Dictionary containing roadmap data with exactly 10 places
    """
    logger.info(f"Generating roadmap for user {user_id}")
    
    # Get user travel preferences
    travel_prefs = get_user_travel_preferences(user_id)
    
    if not travel_prefs:
        logger.warning(f"No travel preferences found for user {user_id}")
        return {"error": "No travel preferences found"}
    
    # Extract preferences
    budget = travel_prefs.get("budget", "medium")
    budget_level = map_budget_level(budget)
    accessibility_needs = travel_prefs.get("accessibility_needs", [])
    group_type = travel_prefs.get("group_type", "")
    travel_dates = travel_prefs.get("travel_dates", "")
    
    # Get preferred destinations (if any)
    destinations = travel_prefs.get("destinations", [])
    
    # Extract month from travel dates
    travel_month = parse_travel_dates(travel_dates)
    
    logger.info(f"User {user_id} preferences: Budget={budget}, Group={group_type}, Month={travel_month}")
    logger.info(f"Accessibility needs: {accessibility_needs}, Destinations: {destinations}")
    
    # Get all possible places
    all_places = list(places_collection.find())
    
    # --- CRITICAL FILTERS PIPELINE ---
    
    # STAGE 1: Accessibility Filter
    if accessibility_needs:
        filtered_places = [
            place for place in all_places
            if check_accessibility_compatibility(place, accessibility_needs)
        ]
        
        # FALLBACK 1: If no places match accessibility, use all places
        if len(filtered_places) == 0:
            logger.warning(f"⚠️ No places match accessibility needs {accessibility_needs}, using all places")
            filtered_places = all_places
    else:
        filtered_places = all_places
    
    logger.info(f"After accessibility filter: {len(filtered_places)}/{len(all_places)} places remaining")
    
    # STAGE 2: Destination Filter (CRITICAL - no fallback)
    if destinations and len(destinations) > 0:
        destination_places = []
        for place in filtered_places:
            location = place.get("location", {})
            city = location.get("city", "")
            country = location.get("country", "")
            
            # Check if place matches any preferred destination
            for destination in destinations:
                if (destination.lower() in city.lower() or 
                    destination.lower() in country.lower() or
                    city.lower() in destination.lower() or
                    country.lower() in destination.lower()):
                    destination_places.append(place)
                    break
        
        # No fallback here - destinations are critical
        filtered_places = destination_places
        logger.info(f"After destination filter: {len(filtered_places)} places in requested destinations")
    
    # --- SCORING PHASE: Score remaining places ---
    scored_places = []
    
    for place in filtered_places:
        # 1. Budget score (30%)
        place_budget = place.get("budget", "medium")
        place_budget_level = map_budget_level(place_budget)
        budget_score = calculate_budget_compatibility(place_budget_level, budget_level)
        
        # 2. Accessibility score (20%)
        place_accessibility = place.get("accessibility", [])
        if not isinstance(place_accessibility, list):
            place_accessibility = [place_accessibility] if place_accessibility else []
        
        accessibility_score = len(place_accessibility) / 10  # Normalize assuming max 10 features
        
        # 3. Group type score (30%)
        group_score = 0.5  # Default score
        place_group_type = place.get("group_type", "")
        
        if group_type and place_group_type:
            if group_type.lower() == place_group_type.lower():
                group_score = 1.0  # Exact match
            elif group_type.lower() in place_group_type.lower() or place_group_type.lower() in group_type.lower():
                group_score = 0.8  # Partial match
        
        # 4. Seasonal score (20%)
        seasonal_score = 0.5  # Default score
        appropriate_time = place.get("appropriate_time", [])
        
        if travel_month and appropriate_time and isinstance(appropriate_time, list):
            if travel_month in appropriate_time:
                seasonal_score = 1.0  # Direct month match
        
        # Calculate total weighted score
        total_score = (
            budget_score * 0.3 +
            accessibility_score * 0.2 +
            group_score * 0.3 +
            seasonal_score * 0.2
        )
        
        # Add to scored places
        scored_places.append({
            "place": place,
            "score": total_score,
            "budget_score": budget_score,
            "accessibility_score": accessibility_score,
            "group_score": group_score,
            "seasonal_score": seasonal_score
        })
    
    # Sort places by score (descending)
    scored_places.sort(key=lambda x: x["score"], reverse=True)
    
    # Prepare final roadmap
    roadmap = {
        "start_date": travel_dates,
        "budget_level": budget_level,
        "group_type": group_type,
        "places": [],
        "routes": [],
        "accessibility_needs": accessibility_needs,
        "seasonal_weights": {travel_month: 1.0} if travel_month else {}
    }
    
    # Add top scoring places to roadmap (up to 10)
    for item in scored_places[:10]:
        place = item["place"]
        
        # Add score details for explanation
        place["match_scores"] = {
            "total": item["score"],
            "budget": item["budget_score"],
            "accessibility": item["accessibility_score"],
            "group": item["group_score"],
            "seasonal": item["seasonal_score"]
        }
        
        roadmap["places"].append(place)
    
    # --- FALLBACK MECHANISMS: Ensure we have exactly 10 places ---
    
    # FALLBACK 1: If we don't have 10 places, add nearby places based on location
    if len(roadmap["places"]) < 10:
        needed_places = 10 - len(roadmap["places"])
        logger.info(f"Need {needed_places} more places for roadmap, adding nearby places")
        
        nearby_places = []
        
        # Get locations of current places
        current_locations = []
        for place in roadmap["places"]:
            loc = place.get("location", {})
            lat = loc.get("latitude")
            lng = loc.get("longitude")
            if lat is not None and lng is not None:
                current_locations.append((lat, lng, place["_id"]))
        
        # If no locations available, skip this step
        if current_locations:
            # For each place in our database, check proximity to current places
            for place in all_places:
                # Skip if already in roadmap
                if place["_id"] in [p["_id"] for p in roadmap["places"]]:
                    continue
                
                # Check location proximity
                loc = place.get("location", {})
                lat = loc.get("latitude")
                lng = loc.get("longitude")
                
                if lat is not None and lng is not None:
                    # Find minimum distance to any current place
                    min_distance = float('inf')
                    for curr_lat, curr_lng, curr_id in current_locations:
                        try:
                            distance = geodesic((lat, lng), (curr_lat, curr_lng)).kilometers
                            min_distance = min(min_distance, distance)
                        except Exception:
                            continue
                    
                    # If within reasonable distance (100km), add to nearby places
                    if min_distance < 100:
                        nearby_places.append({
                            "place": place,
                            "distance": min_distance
                        })
            
            # Sort by distance and add to roadmap
            nearby_places.sort(key=lambda x: x["distance"])
            
            for item in nearby_places[:needed_places]:
                place = item["place"]
                place["match_scores"] = {
                    "total": 0.4,  # Lower score than primary matches
                    "budget": 0.5,
                    "accessibility": 0.5,
                    "group": 0.5,
                    "seasonal": 0.5,
                    "source": "nearby"
                }
                roadmap["places"].append(place)
                logger.info(f"Added nearby place: {place.get('name')} ({item['distance']:.1f}km)")
    
    # FALLBACK 2: If still not enough, add trending places
    if len(roadmap["places"]) < 10:
        needed_places = 10 - len(roadmap["places"])
        logger.info(f"Need {needed_places} more places for roadmap, adding trending places")
        
        # Get recent interactions to find trending places
        recent_date = datetime.now() - timedelta(days=14)
        trending_interactions = list(interactions_collection.aggregate([
            {"$match": {"timestamp": {"$gte": recent_date}}},
            {"$group": {"_id": "$place_id", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": needed_places * 2}
        ]))
        
        trending_place_ids = [item["_id"] for item in trending_interactions]
        
        # Filter out places already in roadmap
        trending_place_ids = [pid for pid in trending_place_ids 
                             if pid not in [p["_id"] for p in roadmap["places"]]]
        
        # Get place details
        if trending_place_ids:
            trending_places = list(places_collection.find({"_id": {"$in": trending_place_ids}}))
            
            for place in trending_places[:needed_places]:
                place["match_scores"] = {
                    "total": 0.3,  # Lower score than other matches
                    "budget": 0.5,
                    "accessibility": 0.5,
                    "group": 0.5,
                    "seasonal": 0.5,
                    "source": "trending"
                }
                roadmap["places"].append(place)
                logger.info(f"Added trending place: {place.get('name')}")
    
    # FALLBACK 3: Final fallback - add any top-rated places
    if len(roadmap["places"]) < 10:
        needed_places = 10 - len(roadmap["places"])
        logger.info(f"Need {needed_places} more places for roadmap, adding top-rated places")
        
        # Exclude places already in roadmap
        current_place_ids = [p["_id"] for p in roadmap["places"]]
        top_places = list(places_collection.find(
            {"_id": {"$nin": current_place_ids}}
        ).sort("average_rating", -1).limit(needed_places))
        
        for place in top_places:
            place["match_scores"] = {
                "total": 0.2,  # Lowest score for fallback
                "budget": 0.5,
                "accessibility": 0.5,
                "group": 0.5,
                "seasonal": 0.5,
                "source": "fallback"
            }
            roadmap["places"].append(place)
            logger.info(f"Added fallback place: {place.get('name')}")
    
    # Ensure we have exactly 10 places (trim if somehow more)
    roadmap["places"] = roadmap["places"][:10]
    
    # Add routes between places
    if len(roadmap["places"]) >= 2:
        for i in range(len(roadmap["places"]) - 1):
            place1 = roadmap["places"][i]
            place2 = roadmap["places"][i + 1]
            
            roadmap["routes"].append({
                "from": place1["_id"],
                "to": place2["_id"],
                "from_name": place1.get("name", ""),
                "to_name": place2.get("name", ""),
                "type": "direct"
            })
    
    logger.info(f"Generated roadmap with {len(roadmap['places'])} places and {len(roadmap['routes'])} routes")
    return roadmap
def simplify_roadmap_to_list(roadmap_data):
    """
    Simplify the roadmap data to a flat list format for easier consumption.
    
    Args:
        roadmap_data: Original roadmap data
        
    Returns:
        List of places with route information
    """
    if not roadmap_data or "places" not in roadmap_data:
        return []
    
    places = roadmap_data.get("places", [])
    routes = roadmap_data.get("routes", [])
    
    # Create a mapping from place ID to route
    next_stops = {}
    for route in routes:
        from_id = route.get("from")
        if from_id:
            next_stops[from_id] = route
    
    # Create the simplified list
    simplified = []
    
    for place in places:
        place_id = place.get("_id")
        
        # Get the route to the next place if available
        next_route = next_stops.get(place_id, {})
        
        # Create simplified entry
        entry = {
            "place": place,
            "next_destination": next_route.get("to_name") if next_route else None,
            "next_id": next_route.get("to") if next_route else None,
            "match_scores": place.get("match_scores", {})
        }
        
        simplified.append(entry)
    
    return simplified

async def get_roadmap_with_caching(user_id):
    """
    Get a roadmap for a user with caching.
    Only regenerates if user preferences have changed since last generation.
    
    Args:
        user_id: User ID
        
    Returns:
        Roadmap data
    """
    try:
        # Check for cached roadmap
        cached_roadmap = roadmaps_collection.find_one({"user_id": user_id})
        
        if cached_roadmap:
            logger.info(f"Found cached roadmap for user {user_id}")
            
            # Check if user preferences have changed
            current_prefs = get_user_travel_preferences(user_id)
            cached_prefs = cached_roadmap.get("travel_preferences")
            
            if current_prefs and cached_prefs:
                # Compare only the preference fields that affect roadmap generation
                preferences_changed = (
                    current_prefs.get("budget") != cached_prefs.get("budget") or
                    current_prefs.get("accessibility_needs") != cached_prefs.get("accessibility_needs") or
                    current_prefs.get("group_type") != cached_prefs.get("group_type") or
                    current_prefs.get("travel_dates") != cached_prefs.get("travel_dates") or
                    current_prefs.get("destinations") != cached_prefs.get("destinations")
                )
                
                if not preferences_changed:
                    logger.info(f"Using cached roadmap for user {user_id} (preferences unchanged)")
                    return cached_roadmap.get("roadmap_data")
                
                logger.info(f"Regenerating roadmap for user {user_id} (preferences changed)")
        
        # Generate new roadmap
        roadmap_data = generate_hybrid_roadmap(user_id)
        
        # Cache the new roadmap
        now = datetime.now()
        travel_prefs = get_user_travel_preferences(user_id)
        
        roadmaps_collection.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "user_id": user_id,
                    "roadmap_data": roadmap_data,
                    "travel_preferences": travel_prefs,
                    "created_at": now
                }
            },
            upsert=True
        )
        
        return roadmap_data
        
    except Exception as e:
        logger.error(f"Error getting roadmap with caching: {str(e)}")
        # Fallback to generating a new roadmap without caching
        return generate_hybrid_roadmap(user_id)
# --- PART 6: API ENDPOINTS (RECOMMENDATIONS) ---

@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {
        "success": True,
        "message": "Travel API v2.0.0 is running",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connectivity
        db_status = "connected"
        try:
            # Ping the database
            client.admin.command('ping')
        except Exception as e:
            db_status = f"disconnected: {str(e)}"
            
        # Check NLP model status
        nlp_status = "loaded"
        nlp_type = getattr(nlp, 'name', type(nlp).__name__)
        
        # Test if vectors are working
        test_doc = nlp("travel")
        has_vectors = hasattr(test_doc, 'vector_norm') and test_doc.vector_norm > 0
        
        if not has_vectors:
            nlp_status = "loaded without vectors"
            
        return {
            "success": True,
            "status": "healthy",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "database": db_status,
            "nlp_model": {
                "status": nlp_status,
                "type": nlp_type,
                "has_vectors": has_vectors
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "status": "unhealthy",
                "error": str(e)
            }
        )

@app.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: str,
    num: int = Query(10, ge=1, le=50),
    force_refresh: bool = Query(False),
    background_tasks: BackgroundTasks = None
):
    """
    Get recommendations for a user with progressive pagination.
    First request: Return 10 new places
    Second request: Return 10 new places + previous 10 = 20 total
    Third request and beyond: Return 10 new places + 20 most recent shown places = 30 total
    
    Args:
        user_id: User ID
        num: Number of NEW recommendations to return (default 10)
        force_refresh: Whether to force fresh recommendations
    """
    try:
        # Get recommendations with the enhanced progressive pagination
        recommendation_data = get_recommendations_with_caching(
            user_id, 
            force_refresh=force_refresh, 
            num_new_recommendations=num,  # Number of new recommendations to fetch
            max_total=max(30, num * 3)  # Ensure we have enough capacity for history
        )
        
        # Combine new and previously shown recommendations
        all_recommendations = recommendation_data["new_recommendations"] + recommendation_data["previously_shown"]
        
        # Check if we need to regenerate cache
        if background_tasks:
            cache_count = recommendations_cache_collection.count_documents({"user_id": user_id})
            if cache_count < 4:
                # Schedule cache regeneration in background
                logger.info(f"Cache count low ({cache_count}), scheduling regeneration")
                background_tasks.add_task(background_cache_recommendations, user_id, 6)
        
        # Return recommendations
        return {
            "success": True,
            "user_id": user_id,
            "count": len(all_recommendations),
            "new_count": len(recommendation_data["new_recommendations"]),
            "history_count": len(recommendation_data["previously_shown"]),
            "recommendations": all_recommendations,
            "cache_used": not force_refresh and len(recommendation_data["new_recommendations"]) > 0
        }
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/recommendations")
async def create_recommendations(request: RecommendationRequest, background_tasks: BackgroundTasks):
    """Generate recommendations for a user (force refresh)"""
    try:
        user_id = request.user_id
        num_recommendations = request.num_recommendations
        
        # Generate fresh recommendations
        recommendations = generate_final_recommendations(user_id, num_recommendations)
        
        # Clear existing cache for this user
        recommendations_cache_collection.delete_many({"user_id": user_id})
        
        # Store as sequence 0 (new generation)
        recommendations_cache_collection.insert_one({
            "user_id": user_id,
            "sequence": 0,
            "recommendations": recommendations,
            "timestamp": datetime.now()
        })
        
        # Schedule cache generation for future requests
        background_tasks.add_task(background_cache_recommendations, user_id, 6)
        
        return {
            "success": True,
            "user_id": user_id,
            "count": len(recommendations),
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/cache/generate/{user_id}")
async def force_cache_generation(
    user_id: str, 
    background_tasks: BackgroundTasks,
    num_entries: int = Query(6, ge=1, le=20)
):
    """
    Force cache generation for a user
    
    This is an admin endpoint to trigger cache generation
    """
    # Check if generation is already in progress
    cache_lock_key = f"cache_lock_{user_id}"
    lock = cache_locks_collection.find_one({"_id": cache_lock_key})
    
    if lock:
        # Check if the lock is stale (older than 5 minutes)
        lock_time = lock.get("timestamp", datetime.min)
        if isinstance(lock_time, str):
            try:
                lock_time = datetime.fromisoformat(lock_time.replace('Z', '+00:00'))
            except Exception:
                lock_time = datetime.min
                
        if (datetime.now() - lock_time).total_seconds() < 300:  # 5 minutes
            return {
                "success": True,
                "message": f"Cache generation already in progress for user {user_id}"
            }
    
    # Schedule cache generation
    background_tasks.add_task(background_cache_recommendations, user_id, num_entries)
    
    return {
        "success": True,
        "message": f"Started generation of {num_entries} cache entries for user {user_id}"
    }

@app.get("/cache/status/{user_id}")
async def get_cache_status(user_id: str):
    """
    Get the status of the recommendation cache for a user
    
    Args:
        user_id: User ID to check cache for
    """
    try:
        # Check if lock exists
        cache_lock_key = f"cache_lock_{user_id}"
        lock = cache_locks_collection.find_one({"_id": cache_lock_key})
        
        # Get cached entries
        cached_entries = get_user_cached_recommendations(user_id)
        
        # Format timestamps
        if cached_entries:
            for entry in cached_entries:
                if "timestamp" in entry and not isinstance(entry["timestamp"], str):
                    entry["timestamp"] = entry["timestamp"].isoformat()
        
        return {
            "success": True,
            "user_id": user_id,
            "generation_in_progress": lock is not None,
            "cache_count": len(cached_entries),
            "cache_entries": [
                {
                    "sequence": entry.get("sequence"),
                    "timestamp": entry.get("timestamp")
                }
                for entry in cached_entries
            ]
        }
    except Exception as e:
        logger.error(f"Error getting cache status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.delete("/cache/{user_id}")
async def clear_cache(user_id: str):
    """
    Clear the recommendation cache for a user
    
    Args:
        user_id: User ID to clear cache for
    """
    try:
        # Delete all cache entries
        result = recommendations_cache_collection.delete_many({"user_id": user_id})
        deleted_count = result.deleted_count
        
        # Also clear locks
        lock_result = cache_locks_collection.delete_one({"_id": f"cache_lock_{user_id}"})
        
        return {
            "success": True,
            "user_id": user_id,
            "deleted_count": deleted_count,
            "lock_cleared": lock_result.deleted_count > 0,
            "message": f"Cleared {deleted_count} cache entries for user {user_id}"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.delete("/shown-places/{user_id}")
async def reset_shown_places(user_id: str):
    """Reset all shown places for a user"""
    try:
        result = shown_places_collection.delete_many({"user_id": user_id})
        deleted_count = result.deleted_count
        
        return {
            "success": True,
            "user_id": user_id,
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} shown place records for user {user_id}"
        }
    except Exception as e:
        logger.error(f"Error resetting shown places: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/shown-places/{user_id}")
async def get_user_shown_places(user_id: str):
    """
    Get places that have been shown to a user
    
    Args:
        user_id: User ID
    """
    try:
        # Get previously shown places
        shown_place_ids = get_previously_shown_places(user_id)
        last_shown_place_ids = get_last_shown_places(user_id)
        
        # Get the actual place documents for the last shown
        last_shown_places = []
        if last_shown_place_ids:
            last_shown_places = list(places_collection.find({"_id": {"$in": last_shown_place_ids}}))
        
        return {
            "success": True,
            "user_id": user_id,
            "all_shown_count": len(shown_place_ids),
            "all_shown_ids": shown_place_ids,
            "last_shown_count": len(last_shown_place_ids),
            "last_shown_ids": last_shown_place_ids,
            "last_shown_places": last_shown_places
        }
    except Exception as e:
        logger.error(f"Error getting shown places: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
# --- PART 7: API ENDPOINTS (SEARCH AND ROADMAP) ---

@app.get("/search/{user_id}")
async def search_places(
    user_id: str,
    query: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50)
):
    """
    Search for places based on a text query
    
    Args:
        user_id: User ID (for tracking)
        query: Search query string
        limit: Maximum number of results to return
    """
    try:
        # Track search query
        search_queries_collection.insert_one({
            "user_id": user_id,
            "query": query,
            "timestamp": datetime.now()
        })
        
        # Get all places
        all_places = list(places_collection.find())
        
        # Improved method to score results with semantic search
        results = []
        
        # Check if NLP model has word vectors
        test_doc = nlp("test")
        has_vectors = hasattr(test_doc, 'vector_norm') and test_doc.vector_norm > 0
        
        if has_vectors:
            # Use semantic search for better matching
            query_doc = nlp(query.lower())
            
            for place in all_places:
                # Initialize score components
                name_score = 0
                desc_score = 0
                tag_score = 0
                category_score = 0
                
                # 1. Exact name match (highest weight)
                if query.lower() in place.get("name", "").lower():
                    name_score = 0.9  # Direct substring match
                
                # Try semantic match on name
                place_name = place.get("name", "")
                if place_name:
                    place_name_doc = nlp(place_name.lower())
                    name_similarity = query_doc.similarity(place_name_doc)
                    name_score = max(name_score, name_similarity * 0.8)  # Max of exact or semantic
                
                # 2. Description match
                description = place.get("description", "")
                if description:
                    if query.lower() in description.lower():
                        desc_score = 0.5  # Direct substring match in description
                    
                    # Semantic similarity for description (if not too long)
                    if len(description) < 1000:  # Avoid processing very long descriptions
                        desc_doc = nlp(description.lower())
                        desc_score = max(desc_score, query_doc.similarity(desc_doc) * 0.5)
                
                # 3. Tags match
                tags = place.get("tags", [])
                if tags:
                    # Check for direct tag matches
                    if query.lower() in [tag.lower() for tag in tags]:
                        tag_score = 0.8  # Direct tag match
                    
                    # Semantic similarity for tags
                    max_tag_similarity = 0
                    for tag in tags:
                        tag_doc = nlp(tag.lower())
                        similarity = query_doc.similarity(tag_doc)
                        max_tag_similarity = max(max_tag_similarity, similarity)
                    
                    tag_score = max(tag_score, max_tag_similarity * 0.7)
                
                # 4. Category match
                category = place.get("category", "")
                if category:
                    if query.lower() in category.lower():
                        category_score = 0.7  # Direct category match
                    
                    # Semantic similarity for category
                    category_doc = nlp(category.lower())
                    category_score = max(category_score, query_doc.similarity(category_doc) * 0.6)
                
                # Compute final score with weights
                # Name (40%), Tags (30%), Category (20%), Description (10%)
                final_score = (
                    0.4 * name_score +
                    0.3 * tag_score +
                    0.2 * category_score +
                    0.1 * desc_score
                )
                
                # Only add results with a minimum relevance
                if final_score > 0.3:  # Threshold for relevance
                    place["search_score"] = final_score
                    results.append({
                        "place": place,
                        "score": final_score
                    })
                
        else:
            # Fallback to basic text matching if vectors aren't available
            logger.warning("Word vectors not available, using basic text search")
            
            for place in all_places:
                score = 0
                
                # Exact name match - highest score
                if query.lower() in place.get("name", "").lower():
                    score = 1.0
                # Tag match - high score
                elif "tags" in place and any(query.lower() in tag.lower() for tag in place.get("tags", [])):
                    score = 0.8
                # Category match - medium score
                elif "category" in place and query.lower() in place.get("category", "").lower():
                    score = 0.7
                # Description match - lower score
                elif "description" in place and query.lower() in place.get("description", "").lower():
                    score = 0.5
                    
                if score > 0:
                    # Add to results with score
                    place["search_score"] = score
                    results.append({
                        "place": place,
                        "score": score
                    })
                
        # Sort by score (highest first) and limit results
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)[:limit]
        
        # Extract just the place data
        final_results = [item["place"] for item in sorted_results]
        
        return {
            "success": True,
            "user_id": user_id,
            "query": query,
            "count": len(final_results),
            "results": final_results
        }
        
    except Exception as e:
        logger.error(f"Error searching places: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/search-history/{user_id}")
async def get_search_history(
    user_id: str,
    limit: int = Query(10, ge=1, le=50)
):
    """
    Get search history for a user
    
    Args:
        user_id: User ID
        limit: Maximum number of results to return
    """
    try:
        # Get search history sorted by newest first
        history = list(
            search_queries_collection.find(
                {"user_id": user_id},
                {"_id": 0, "user_id": 1, "query": 1, "timestamp": 1}
            )
            .sort("timestamp", -1)
            .limit(limit)
        )
        
        # Format timestamps
        for item in history:
            if "timestamp" in item and not isinstance(item["timestamp"], str):
                item["timestamp"] = item["timestamp"].isoformat()
        
        return {
            "success": True,
            "user_id": user_id,
            "count": len(history),
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Error getting search history: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.delete("/search-history/{user_id}")
async def clear_search_history(user_id: str):
    """
    Clear search history for a user
    
    Args:
        user_id: User ID
    """
    try:
        result = search_queries_collection.delete_many({"user_id": user_id})
        deleted_count = result.deleted_count
        
        return {
            "success": True,
            "user_id": user_id,
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} search history records for user {user_id}"
        }
    except Exception as e:
        logger.error(f"Error clearing search history: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# --- Roadmap API Endpoints ---

@app.get("/roadmap/{user_id}")
async def get_roadmap(user_id: str):
    """
    Get a travel roadmap for a specific user
    
    Uses caching: Only regenerates if user's travel preferences have changed
    """
    try:
        logger.info(f"Roadmap request for user {user_id}")
        
        # Get roadmap (cached or newly generated)
        roadmap_data = await get_roadmap_with_caching(user_id)
        
        # Simplify to list format
        simplified_list = simplify_roadmap_to_list(roadmap_data)
        
        return {
            "success": True, 
            "user_id": user_id, 
            "count": len(simplified_list),
            "data": simplified_list,
            "metadata": {
                "budget_level": roadmap_data.get("budget_level"),
                "group_type": roadmap_data.get("group_type"),
                "start_date": roadmap_data.get("start_date"),
                "accessibility_needs": roadmap_data.get("accessibility_needs", [])
            }
        }
    except Exception as e:
        logger.error(f"Error generating roadmap: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/roadmap")
async def create_roadmap(request: RoadmapRequest):
    """
    Generate a new travel roadmap for a user (force regeneration)
    """
    try:
        user_id = request.user_id
        logger.info(f"Force regenerating roadmap for user {user_id}")
        
        # Force generation of new roadmap
        roadmap_data = generate_hybrid_roadmap(user_id)
        
        # Store in cache
        now = datetime.now()
        travel_prefs = get_user_travel_preferences(user_id)
        roadmaps_collection.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "user_id": user_id,
                    "roadmap_data": roadmap_data,
                    "travel_preferences": travel_prefs,
                    "created_at": now
                }
            },
            upsert=True
        )
        
        # Simplify to list format
        simplified_list = simplify_roadmap_to_list(roadmap_data)
        
        return {
            "success": True, 
            "user_id": user_id, 
            "count": len(simplified_list),
            "data": simplified_list,
            "metadata": {
                "budget_level": roadmap_data.get("budget_level"),
                "group_type": roadmap_data.get("group_type"),
                "start_date": roadmap_data.get("start_date"),
                "accessibility_needs": roadmap_data.get("accessibility_needs", [])
            }
        }
    except Exception as e:
        logger.error(f"Error generating roadmap: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/roadmap-status/{user_id}")
async def get_roadmap_cache_status(user_id: str):
    """
    Get the status of the roadmap cache for a user
    
    Args:
        user_id: User ID
    """
    try:
        # Check if roadmap exists in cache
        cached_roadmap = roadmaps_collection.find_one({"user_id": user_id})
        
        if cached_roadmap:
            # Format created_at timestamp
            created_at = cached_roadmap.get("created_at")
            if created_at and not isinstance(created_at, str):
                created_at = created_at.isoformat()
                
            # Get current preferences
            current_prefs = get_user_travel_preferences(user_id)
            cached_prefs = cached_roadmap.get("travel_preferences")
            
            # Compare preferences to check if they've changed
            preferences_changed = True
            if current_prefs and cached_prefs:
                preferences_changed = (
                    current_prefs.get("budget") != cached_prefs.get("budget") or
                    current_prefs.get("accessibility_needs") != cached_prefs.get("accessibility_needs") or
                    current_prefs.get("group_type") != cached_prefs.get("group_type") or
                    current_prefs.get("travel_dates") != cached_prefs.get("travel_dates") or
                    current_prefs.get("destinations") != cached_prefs.get("destinations")
                )
            
            return {
                "success": True,
                "user_id": user_id,
                "cache_exists": True,
                "created_at": created_at,
                "place_count": len(cached_roadmap.get("roadmap_data", {}).get("places", [])),
                "preferences_changed": preferences_changed,
                "cached_preferences": cached_prefs,
                "current_preferences": current_prefs
            }
        else:
            return {
                "success": True,
                "user_id": user_id,
                "cache_exists": False
            }
    except Exception as e:
        logger.error(f"Error getting roadmap cache status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.delete("/roadmap/{user_id}")
async def clear_roadmap_cache(user_id: str):
    """
    Clear the roadmap cache for a specific user
    """
    try:
        result = roadmaps_collection.delete_one({"user_id": user_id})
        deleted = result.deleted_count > 0
        
        return {
            "success": True,
            "user_id": user_id,
            "cache_cleared": deleted
        }
    except Exception as e:
        logger.error(f"Error clearing roadmap cache: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
# --- PART 8: ERROR HANDLERS AND SERVER STARTUP ---

# --- Place and User Endpoints ---

@app.get("/place/{place_id}")
async def get_place(place_id: str):
    """
    Get details for a specific place
    
    Args:
        place_id: Place ID
    """
    try:
        place = places_collection.find_one({"_id": place_id})
        
        if not place:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": f"Place {place_id} not found"}
            )
            
        return {
            "success": True,
            "place": place
        }
    except Exception as e:
        logger.error(f"Error getting place details: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/places/stats")
async def get_places_stats():
    """Get statistics about places in the database"""
    try:
        total_count = places_collection.count_documents({})
        
        # Get category distribution
        category_pipeline = [
            {"$group": {"_id": "$category", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        categories = list(places_collection.aggregate(category_pipeline))
        
        # Get average rating
        rating_pipeline = [
            {"$match": {"average_rating": {"$exists": True}}},
            {"$group": {"_id": None, "avg": {"$avg": "$average_rating"}}}
        ]
        avg_rating_result = list(places_collection.aggregate(rating_pipeline))
        avg_rating = avg_rating_result[0]["avg"] if avg_rating_result else 0
        
        return {
            "success": True,
            "total_places": total_count,
            "categories": categories,
            "average_rating": avg_rating
        }
    except Exception as e:
        logger.error(f"Error getting places stats: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/user-preferences/{user_id}")
async def get_user_preferences_endpoint(user_id: str):
    """
    Get user preferences
    
    Args:
        user_id: User ID
    """
    try:
        # Get general preferences
        preferences = get_user_preferences(user_id)
        
        # Get travel preferences
        travel_prefs = get_user_travel_preferences(user_id)
        
        if not preferences and not travel_prefs:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": f"No preferences found for user {user_id}"}
            )
            
        return {
            "success": True,
            "user_id": user_id,
            "general_preferences": preferences,
            "travel_preferences": travel_prefs
        }
    except Exception as e:
        logger.error(f"Error getting user preferences: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/system/stats")
async def get_system_stats():
    """Get system statistics for monitoring"""
    try:
        # Get collection counts
        collection_stats = {}
        
        for collection_name in [
            "users", "places", "interactions", "search_queries", 
            "user_travel_preferences", "recommendations_cache", 
            "shown_places", "roadmaps", "cache_locks"
        ]:
            collection_stats[collection_name] = db[collection_name].count_documents({})
            
        # Get NLP model info
        nlp_info = {
            "name": getattr(nlp, "name", str(type(nlp).__name__)),
            "has_vectors": nlp.vocab.vectors.n_keys > 0
        }
        
        # Memory usage of recent cache entries (sample)
        cache_sample = list(recommendations_cache_collection.find().limit(1))
        cache_size = len(str(cache_sample)) if cache_sample else 0
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "collection_stats": collection_stats,
            "nlp_model": nlp_info,
            "cache_sample_size_bytes": cache_size,
            "api_version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/debug/nlp-test")
async def test_nlp():
    """Test NLP model with word similarity comparisons"""
    try:
        # Check if model has vectors
        test_word = "travel"
        test_doc = nlp(test_word)
        has_vectors = hasattr(test_doc, 'vector_norm') and test_doc.vector_norm > 0
        
        if not has_vectors:
            return {
                "success": False,
                "error": "NLP model does not have word vectors",
                "model": getattr(nlp, "name", str(type(nlp).__name__)),
                "fallback_active": isinstance(nlp, type) and nlp.__name__ == "DummyNLP"
            }
            
        # Test word similarity pairs
        word_pairs = [
            ("beach", "ocean"),
            ("mountain", "hiking"),
            ("museum", "history"),
            ("restaurant", "food"),
            ("hotel", "accommodation")
        ]
        
        similarity_results = {}
        for word1, word2 in word_pairs:
            doc1 = nlp(word1)
            doc2 = nlp(word2)
            similarity = doc1.similarity(doc2)
            similarity_results[f"{word1}_{word2}"] = similarity
            
        return {
            "success": True,
            "has_vectors": has_vectors,
            "model": getattr(nlp, "name", str(type(nlp).__name__)),
            "similarity_results": similarity_results,
            "vector_examples": {
                "travel": test_doc.vector[:5].tolist(),  # First 5 dimensions
                "norm": float(test_doc.vector_norm)
            }
        }
    except Exception as e:
        logger.error(f"Error testing NLP model: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# --- Error Handlers ---

@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"success": False, "error": "Resource not found"}
    )

@app.exception_handler(500)
async def server_exception_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"}
    )

# --- Server Startup ---
if __name__ == "__main__":
    import uvicorn
    
    # Check environment for port
    port = int(os.environ.get("PORT", 8000))
    
    # Log startup information
    logger.info("=" * 50)
    logger.info("Starting Travel API Server v2.0.0")
    logger.info(f"Using port: {port}")
    
    # Check MongoDB connection
    try:
        client.server_info()
        logger.info("✅ MongoDB connection verified")
        
        # Log collection counts
        for coll_name in ["users", "places", "recommendations_cache"]:
            count = db[coll_name].count_documents({})
            logger.info(f"Collection {coll_name}: {count} documents")
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        
    # Log NLP model status
    has_vectors = nlp.vocab.vectors.n_keys > 0
    logger.info(f"NLP Model: {getattr(nlp, 'name', type(nlp).__name__)}")
    logger.info(f"Word Vectors: {'Available' if has_vectors else 'NOT AVAILABLE'}")
    logger.info("=" * 50)
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=port)
