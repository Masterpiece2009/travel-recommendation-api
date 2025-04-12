# --- PART 1: IMPORTS AND CONFIGURATION ---

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

    # Return dummy NLP object as fallback with clear logging
    class DummyNLP:
        def __init__(self):
            self.name = "DummyNLP-Fallback"
            self.vocab = type('obj', (object,), {
                'vectors': type('obj', (object,), {'n_keys': 0})
            })
            logger.critical("⛔ CRITICAL: Using dummy NLP model! Semantic search will NOT work properly.")
            
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

# --- Recommendation System: Core Functions ---

def get_candidate_places(user_preferences, user_id, size=30):
    """
    Get candidate places for recommendations based on user preferences.
    Enhanced with improved semantic search for matching places to user preferences.
    
    Args:
        user_preferences: Dictionary containing preferred_categories and preferred_tags
        user_id: User ID for fetching search history and interactions
        size: Maximum number of candidate places to return
        
    Returns:
        List of candidate places
    """
    if not user_preferences:
        logger.warning(f"No user preferences found for user {user_id}, returning popular places")
        return list(places_collection.find().sort([("average_rating", -1)]).limit(size))
    
    # Extract user preferences
    preferred_categories = user_preferences.get("preferred_categories", [])
    preferred_tags = user_preferences.get("preferred_tags", [])
    
    # --- PART 1: CATEGORY AND TAG MATCHING (60%) ---
    query = {"$or": []}
    
    # Add category filter if we have preferred categories
    if preferred_categories:
        query["$or"].append({"category": {"$in": preferred_categories}})
    
    # Add tags filter if we have preferred tags
    if preferred_tags:
        query["$or"].append({"tags": {"$in": preferred_tags}})
    
    # If we have no preferences to query on, return popular places
    if not query["$or"]:
        logger.info(f"No category or tag preferences for user {user_id}, using popularity")
        return list(places_collection.find().sort([("average_rating", -1)]).limit(size))
    
    # Get places matching categories or tags
    category_tag_places = list(places_collection.find(query).limit(size))
    logger.info(f"Found {len(category_tag_places)} places matching categories/tags for user {user_id}")
    
    # --- PART 2: SEMANTIC SEARCH BASED ON RECENT QUERIES (40%) ---
    # Check if NLP model has word vectors
    test_doc = nlp("test")
    has_vectors = hasattr(test_doc, 'vector_norm') and test_doc.vector_norm > 0
    
    semantic_places = []
    if has_vectors:
        try:
            # Fetch recent search queries for this user
            search_queries = list(search_queries_collection.find(
                {"user_id": user_id}
            ).sort("timestamp", -1).limit(5))
            
            # Extract keywords from search queries
            search_keywords = set()
            for query_doc in search_queries:
                query_text = query_doc.get("query", "")
                if query_text:
                    # Simple keyword extraction
                    stopwords = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
                    keywords = [word.lower() for word in query_text.split() if word.lower() not in stopwords]
                    for keyword in keywords:
                        if len(keyword) > 2:  # Skip very short words
                            search_keywords.add(keyword)
            
            logger.info(f"Extracted {len(search_keywords)} search keywords for user {user_id}")
            
            if search_keywords:
                # Get all places for semantic matching
                all_places = list(places_collection.find())
                
                # Calculate semantic similarity scores
                keyword_place_scores = {}
                match_count = 0
                
                for keyword in search_keywords:
                    keyword_doc = nlp(keyword.lower())
                    
                    for place in all_places:
                        place_id = place["_id"]
                        
                        if place_id not in keyword_place_scores:
                            keyword_place_scores[place_id] = 0
                        
                        # Check each tag for similarity
                        if "tags" in place and isinstance(place["tags"], list):
                            for tag in place["tags"]:
                                tag_doc = nlp(tag.lower())
                                similarity = keyword_doc.similarity(tag_doc)
                                
                                # Add score if similarity is above threshold
                                if similarity > 0.6:  # Threshold for semantic match
                                    keyword_place_scores[place_id] += similarity
                                    match_count += 1
                
                logger.info(f"Found {match_count} semantic matches above threshold 0.6 for user {user_id}")
                
                # Get top matching places
                semantic_matches = [(place_id, score) for place_id, score in keyword_place_scores.items() if score > 0]
                semantic_matches.sort(key=lambda x: x[1], reverse=True)
                
                # Get the actual place documents
                if semantic_matches:
                    matched_ids = [match[0] for match in semantic_matches[:size]]
                    semantic_places = list(places_collection.find({"_id": {"$in": matched_ids}}))
                    
                    # Sort them by score
                    id_to_place = {place["_id"]: place for place in semantic_places}
                    semantic_places = [id_to_place[match_id] for match_id, _ in semantic_matches 
                                      if match_id in id_to_place]
                    
                    logger.info(f"Found {len(semantic_places)} places via semantic search for user {user_id}")
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            # If semantic search fails, we'll just use the category/tag results
    else:
        logger.warning("Word vectors not available, skipping semantic search")
    
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
    
    # If we don't have enough candidates, add some random places
    if len(candidate_places) < size:
        additional_places = list(
            places_collection.find({"_id": {"$nin": list(added_ids)}})
            .sort([("average_rating", -1)])
            .limit(size - len(candidate_places))
        )
        candidate_places.extend(additional_places)
        logger.info(f"Added {len(additional_places)} additional places to reach target size")
    
    logger.info(f"Returning {len(candidate_places)} total candidate places for user {user_id}")
    return candidate_places

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

def update_shown_places(user_id, new_place_ids, max_places=None):
    """
    Update the list of shown places for a user.
    If max_places is provided, limit the list to that many places.
    Also updates last_shown_places for tracking only the most recent request.
    Includes timestamp for TTL (6-hour expiration).
    
    Args:
        user_id: User ID
        new_place_ids: List of place IDs shown in current request
        max_places: Maximum number of places to track (if None, tracks all)
        
    Returns:
        List of all place IDs shown to this user
    """
    try:
        if not new_place_ids:  # Skip update if no new places
            return get_previously_shown_places(user_id)

        existing_places = get_previously_shown_places(user_id)

        # Add new places that aren't already in the list
        all_place_ids = existing_places + [pid for pid in new_place_ids if pid not in existing_places]

        # If max_places is set, limit the list to the most recent places
        if max_places and len(all_place_ids) > max_places:
            all_place_ids = all_place_ids[-max_places:]

        # Update the database with all shown places and timestamp for TTL
        shown_places_collection.update_one(
            {"user_id": user_id},
            {"$set": {
                "place_ids": all_place_ids,
                "last_shown_place_ids": new_place_ids,  # Track only this request's places
                "timestamp": datetime.now()  # Add timestamp for TTL expiration
            }},
            upsert=True
        )

        return all_place_ids
    except Exception as e:
        logger.error(f"Error updating shown places: {e}")
        return []

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

def generate_final_recommendations(user_id, num_recommendations=10):
    """
    Generate the final recommendations using cached shown places and generating new ones as needed.
    
    Args:
        user_id: User ID
        num_recommendations: Number of recommendations to return
        
    Returns:
        List of recommendation objects
    """
    try:
        logger.info(f"Generating recommendations for user {user_id}")
        
        # Get user preferences
        user_preferences = get_user_preferences(user_id)
        
        if not user_preferences:
            logger.warning(f"No preferences found for user {user_id}, using defaults")
            user_preferences = {"preferred_categories": [], "preferred_tags": []}
            
        # Get previously shown place IDs and last shown place IDs
        previously_shown_ids = get_previously_shown_places(user_id)
        last_shown_ids = get_last_shown_places(user_id)
        
        logger.info(f"User {user_id} has {len(previously_shown_ids)} previously shown places, {len(last_shown_ids)} from last request")
        
        recommendations = []
        
        # PART 1: Rerank most recently shown places
        if last_shown_ids:
            # Limit to half of requested recommendations to make room for new places
            refresh_limit = max(3, num_recommendations // 2)
            refreshed_places = refresh_shown_places(user_id, last_shown_ids, refresh_limit)
            
            # Add refreshed places to recommendations
            for place in refreshed_places:
                if len(recommendations) < num_recommendations:
                    place["source"] = "refreshed"
                    recommendations.append(place)
            
            logger.info(f"Added {len(refreshed_places)} refreshed places")
            
        # PART 2: Add new recommendations
        remaining_needed = num_recommendations - len(recommendations)
        
        if remaining_needed > 0:
            # Get candidate places based on user preferences
            candidate_places = get_candidate_places(user_preferences, user_id, size=30)
            
            # Filter out places that have been shown before
            candidate_places = [p for p in candidate_places if p["_id"] not in previously_shown_ids]
            
            # Rank remaining places
            ranked_places = rank_places(candidate_places, user_id)
            
            # Add ranked places up to the limit
            for place in ranked_places:
                if len(recommendations) < num_recommendations:
                    place["source"] = "new"
                    recommendations.append(place)
                else:
                    break
                    
            logger.info(f"Added {min(remaining_needed, len(ranked_places))} new places")
            
        # PART 3: If we still need more recommendations (e.g., if user has seen all places)
        if len(recommendations) < num_recommendations:
            remaining_needed = num_recommendations - len(recommendations)
            logger.info(f"Still need {remaining_needed} more recommendations, considering already shown places")
            
            # Get random places from those already shown (but not from the last request)
            already_shown_excluded_last = [pid for pid in previously_shown_ids if pid not in last_shown_ids]
            
            if already_shown_excluded_last:
                # Get places by ID
                random.shuffle(already_shown_excluded_last)
                place_ids_to_use = already_shown_excluded_last[:remaining_needed]
                reused_places = list(places_collection.find({"_id": {"$in": place_ids_to_use}}))
                
                for place in reused_places:
                    place["source"] = "reused"
                    recommendations.append(place)
                    
                logger.info(f"Added {len(reused_places)} reused places")
                
        # PART 4: Last resort - if still not enough, add any places at all
        if len(recommendations) < num_recommendations:
            remaining_needed = num_recommendations - len(recommendations)
            logger.info(f"Still need {remaining_needed} more recommendations, adding any available places")
            
            last_resort_places = list(places_collection.find().limit(remaining_needed))
            
            for place in last_resort_places:
                if place["_id"] not in [r["_id"] for r in recommendations]:
                    place["source"] = "fallback"
                    recommendations.append(place)
                    
            logger.info(f"Added {len(last_resort_places)} fallback places")
            
        # Track the shown places
        new_place_ids = [place["_id"] for place in recommendations if place["source"] != "refreshed"]
        update_shown_places(user_id, new_place_ids, max_places=100)
        
        logger.info(f"Generated {len(recommendations)} total recommendations for user {user_id}")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        # Return some fallback recommendations in case of error
        fallback_places = list(places_collection.find().sort("average_rating", -1).limit(num_recommendations))
        for place in fallback_places:
            place["source"] = "error_fallback"
        return fallback_places

def get_recommendations_with_caching(user_id, force_refresh=False, num_recommendations=10):
    """
    Get recommendations for a user with caching.
    
    Args:
        user_id: User ID
        force_refresh: Whether to force generation of new recommendations
        num_recommendations: Number of recommendations to return
        
    Returns:
        List of recommendation objects
    """
    try:
        if force_refresh:
            logger.info(f"Force refresh requested for user {user_id}")
            return generate_final_recommendations(user_id, num_recommendations)
            
        # Look for cached recommendations
        cached_entries = get_user_cached_recommendations(user_id)
        
        if not cached_entries:
            logger.info(f"No cached recommendations found for user {user_id}")
            return generate_final_recommendations(user_id, num_recommendations)
            
        # Get the first entry with lowest sequence number
        cached_entry = cached_entries[0]
        
        # Remove the used entry from cache
        recommendations_cache_collection.delete_one({"_id": cached_entry["_id"]})
        
        logger.info(f"Using cached recommendations for user {user_id} (sequence {cached_entry['sequence']})")
        
        # If this was the last entry, schedule background generation of new cache entries
        if len(cached_entries) <= 2:
            logger.info(f"Only {len(cached_entries)} cached entries left for user {user_id}, scheduling more")
            
            # This will be handled by the endpoint using BackgroundTasks
            pass
            
        return cached_entry["recommendations"][:num_recommendations]
        
    except Exception as e:
        logger.error(f"Error getting recommendations with caching: {str(e)}")
        return generate_final_recommendations(user_id, num_recommendations)

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
            # Lock already exists and is held by another process
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
                # Wait a small amount of time between generations for variety
                await asyncio.sleep(0.5)
                
                # Generate fresh recommendations
                recommendations = generate_final_recommendations(user_id, 10)
                
                # Store in cache with incrementing sequence
                sequence = next_sequence + i
                
                # Note the sequence in the stored data
                store_cache_entry(user_id, recommendations, sequence)
                
                logger.info(f"Generated cache entry {i+1}/{num_entries} for user {user_id}")
                
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
    
    Args:
        user_id: User ID
        
    Returns:
        Dictionary containing roadmap data
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
    
    # Get month from travel dates
    travel_month = parse_travel_dates(travel_dates)
    seasonal_weights = get_seasonal_activities(travel_month)
    
    logger.info(f"User {user_id} preferences: Budget={budget}, Group={group_type}, Month={travel_month}")
    
    # Get all possible places
    all_places = list(places_collection.find())
    
    # --- STAGE 1: Apply critical filters ---
    # Filter places that don't meet accessibility requirements
    critical_filtered_places = [
        place for place in all_places
        if check_accessibility_compatibility(place, accessibility_needs)
    ]
    
    logger.info(f"Stage 1: {len(critical_filtered_places)}/{len(all_places)} places passed critical filters")
    
    # --- STAGE 2: Score places based on soft constraints with weights ---
    
    # Define weights for soft constraints:
    # Budget (30%), Accessibility (20%), Group Type (30%), Seasonal (20%)
    constraint_weights = {
        "budget": 0.3,
        "accessibility": 0.2,
        "group_type": 0.3,
        "seasonal": 0.2
    }
    
    scored_places = []
    
    for place in critical_filtered_places:
        # Initialize total score
        total_score = 0
        
        # 1. Budget compatibility (30%)
        place_budget = place.get("budget_level", "medium")
        place_budget_level = map_budget_level(place_budget)
        budget_score = calculate_budget_compatibility(place_budget_level, budget_level)
        
        # 2. Accessibility bonus (20%)
        # Give bonus score for places with more accessibility features than required
        place_features = place.get("accessibility_features", [])
        accessibility_score = len(place_features) / 10  # Normalize assuming max 10 features
        
        # 3. Group type compatibility (30%)
        group_score = 0.5  # Default score
        place_group_types = place.get("suitable_for", [])
        
        if group_type and place_group_types:
            # Check if user's group type matches place's suitable_for
            if isinstance(place_group_types, list) and group_type in place_group_types:
                group_score = 1.0
            elif isinstance(place_group_types, str) and group_type == place_group_types:
                group_score = 1.0
        
        # 4. Seasonal compatibility (20%)
        seasonal_score = 0.5  # Default score
        place_seasons = place.get("best_seasons", [])
        
        if place_seasons:
            # Calculate weighted score based on current season
            season_score = 0
            for season, weight in seasonal_weights.items():
                if season in place_seasons:
                    season_score += weight
            
            seasonal_score = min(1.0, season_score)  # Cap at 1.0
        
        # Calculate weighted total score
        total_score = (
            budget_score * constraint_weights["budget"] +
            accessibility_score * constraint_weights["accessibility"] +
            group_score * constraint_weights["group_type"] +
            seasonal_score * constraint_weights["seasonal"]
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
        "seasonal_weights": seasonal_weights
    }
    
    # Add top scoring places to roadmap
    for item in scored_places[:10]:  # Top 10 places
        place = item["place"]
        
        # Add score details for debugging/explanation
        place["match_scores"] = {
            "total": item["score"],
            "budget": item["budget_score"],
            "accessibility": item["accessibility_score"],
            "group": item["group_score"],
            "seasonal": item["seasonal_score"]
        }
        
        roadmap["places"].append(place)
    
    # Add simple routes between places in order of scoring
    if len(roadmap["places"]) >= 2:
        for i in range(len(roadmap["places"]) - 1):
            place1 = roadmap["places"][i]
            place2 = roadmap["places"][i + 1]
            
            # Calculate simple route (direct line)
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
    force_refresh: bool = Query(False)
):
    """
    Get recommendations for a user
    
    Args:
        user_id: User ID
        num: Number of recommendations to return
        force_refresh: Whether to force fresh recommendations
    """
    try:
        if force_refresh:
            # Generate fresh recommendations
            recommendations = generate_final_recommendations(user_id, num)
        else:
            # Use cached recommendations if available
            recommendations = get_recommendations_with_caching(user_id, force_refresh=False, num_recommendations=num)
        
        return {
            "success": True,
            "user_id": user_id,
            "count": len(recommendations),
            "recommendations": recommendations,
            "cache_used": not force_refresh
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
