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
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Securely Connect to MongoDB
password = os.environ.get("MONGO_PASSWORD", "master2002_B*")  # Fallback for development
encoded_password = urllib.parse.quote_plus(password)

MONGO_URI = f"mongodb+srv://abdelrahman:{encoded_password}@cluster0.goxvb.mongodb.net/travel_app?retryWrites=true&w=majority&appName=Cluster0"
def connect_mongo(uri, retries=3):
    """Attempts to connect to MongoDB with retry logic."""
    for attempt in range(retries):
        try:
            client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.server_info()  # Force connection check
            logger.info("✅ MongoDB connection established!")
            return client
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed (Attempt {attempt + 1}/{retries}): {e}")
    raise Exception("❌ MongoDB connection failed after multiple attempts.")

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
roadmaps_collection = db["roadmaps"]  # New collection for roadmaps

# Create TTL index for roadmaps collection (expires after 24 hours)
try:
    roadmaps_collection.create_index(
        [("created_at", 1)],
        expireAfterSeconds=86400  # 24 hours
    )
    logger.info("Created TTL index on roadmaps collection")
except Exception as e:
    logger.error(f"Error creating TTL index on roadmaps collection: {e}")
    
# --- Initialize spaCy model ---
def load_spacy_model(model="en_core_web_md", retries=2):  # Changed default to md model
    """Attempts to load the spaCy model, downloading it if necessary."""
    logger.info(f"🔄 Attempting to load spaCy model: {model}")
    
    for attempt in range(retries):
        try:
            nlp = spacy.load(model)
            logger.info(f"✅ Successfully loaded spaCy model: {model}")
            return nlp
        except Exception as e:
            logger.error(f"❌ Error loading NLP model (Attempt {attempt + 1}/{retries}): {e}")
            try:
                logger.info(f"📥 Downloading spaCy model: {model}")
                # Use python -m approach which is more reliable in some environments
                import subprocess
                result = subprocess.run([sys.executable, "-m", "spacy", "download", model], 
                                       capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"✅ Successfully downloaded model: {model}")
                else:
                    logger.error(f"❌ Failed to download model: {result.stderr}")
            except Exception as download_err:
                logger.error(f"❌ Failed to download model: {download_err}")

    # Return dummy NLP object as fallback
    class DummyNLP:
        def __init__(self):
            self.name = "DummyNLP-Fallback"
            
        def __call__(self, text):
            class DummyDoc:
                def __init__(self, text):
                    self.text = text
                    self.vector = [0] * 300  # Empty vector
            return DummyDoc(text)
    
    logger.warning("⚠️ CRITICAL: Using dummy NLP model as fallback! Semantic search will NOT work properly.")
    return DummyNLP()

# Try to load the model with word vectors
nlp = load_spacy_model()

# Check if model has word vectors
test_text = "travel"
test_doc = nlp(test_text)
has_vectors = not all(v == 0 for v in test_doc.vector)

if has_vectors:
    logger.info("✅ NLP Model loaded with WORD VECTORS - semantic search will work properly")
else:
    logger.warning("⚠️ NLP Model doesn't have word vectors - semantic search will be LIMITED")

app = FastAPI(
    title="Travel API",
    description="API for travel recommendations and roadmaps",
    version="1.0.0"
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

# --- Shared Utility Functions ---

def get_user_preferences(user_id):
    """Get user general preferences (categories & tags only)"""
    user = users_collection.find_one({"_id": user_id})
    
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
    Falls back to basic word overlap if NLP model doesn't have vectors.
    """
    if not text1 or not text2:
        return 0
        
    try:
        # Try using spaCy word vectors
        doc1 = nlp(text1.lower())
        doc2 = nlp(text2.lower())
        
        # Check if vectors are available (not all models have vectors)
        if doc1.vector_norm and doc2.vector_norm:
            return doc1.similarity(doc2)
        
        # Fall back to basic word overlap if vectors aren't available
        words1 = set(word.lower() for word in text1.split())
        words2 = set(word.lower() for word in text2.split())
        
        if not words1 or not words2:
            return 0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
        
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
    # Convert to datetime object if it's a string
    if isinstance(interaction_time, str):
        try:
            interaction_time = datetime.fromisoformat(interaction_time.replace('Z', '+00:00'))
        except Exception as e:
            logger.error(f"Error parsing timestamp: {e}")
            return weight
            
    # Ensure both datetimes are timezone aware or both are naive
    now = datetime.now()
    
    # If interaction_time is timezone aware, make now timezone aware too
    if hasattr(interaction_time, 'tzinfo') and interaction_time.tzinfo is not None:
        now = datetime.now(timezone.utc)
        
    # Calculate days between now and interaction
    try:
        days_ago = (now - interaction_time).days
        decay = math.exp(-days_ago / 30)  # Exponential decay over 30 days
        return weight * decay
    except Exception as e:
        logger.error(f"Error calculating time decay: {e}")
        return weight  # Return original weight on error
# --- PART 3: RECOMMENDATION ALGORITHM FUNCTIONS ---

# --- Recommendation System: Core Functions ---

def get_candidate_places(user_id, size=30):
    """
    Get candidate places for recommendations based on user preferences.
    Returns a list of places that match user's preferred categories and tags.
    """
    user_prefs = get_user_preferences(user_id)
    if not user_prefs:
        # Fallback if no user preferences found
        return list(places_collection.find().limit(size))
    
    # Extract user preferences
    preferred_categories = user_prefs.get("preferred_categories", [])
    preferred_tags = user_prefs.get("preferred_tags", [])
    
    # --- PART 1: CATEGORY AND TAG MATCHING (60%) ---
    query = {"$or": []}
    
    # Add category filter if we have preferred categories
    if preferred_categories:
        query["$or"].append({"category": {"$in": preferred_categories}})
    
    # Add tags filter if we have preferred tags
    if preferred_tags:
        query["$or"].append({"tags": {"$in": preferred_tags}})
    
    # If we have no preferences to query on, return all places
    if not query["$or"]:
        return list(places_collection.find().limit(size))
        
    # Get places matching categories or tags
    category_tag_places = list(places_collection.find(query).limit(size))
    
    # --- PART 2: SEMANTIC SEARCH BASED ON RECENT QUERIES (40%) ---
    # Only perform semantic search if we have NLP with word vectors
    semantic_places = []
    test_doc = nlp("test")
    has_vectors = not all(v == 0 for v in test_doc.vector)
    
    if has_vectors:
        try:
            # Fetch recent search queries for this user
            search_queries = list(search_queries_collection.find(
                {"user_id": user_id}
            ).sort("timestamp", -1).limit(5))
            
            # Extract keywords from search queries
            search_keywords = set()
            for query in search_queries:
                query_text = query.get("query", "")
                if query_text:
                    # Simple keyword extraction
                    stopwords = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
                    keywords = [word.lower() for word in query_text.split() if word.lower() not in stopwords]
                    for keyword in keywords:
                        search_keywords.add(keyword)
            
            if search_keywords:
                # Get all places for semantic matching
                all_places = list(places_collection.find())
                
                # Calculate semantic similarity scores
                keyword_place_scores = {}
                
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
                                if similarity > 0.6:
                                    keyword_place_scores[place_id] += similarity
                
                # Get top matching places
                semantic_matches = [(place_id, score) for place_id, score in keyword_place_scores.items() if score > 0]
                semantic_matches.sort(key=lambda x: x[1], reverse=True)
                
                # Get the actual place documents
                if semantic_matches:
                    matched_ids = [match[0] for match in semantic_matches]
                    semantic_places = list(places_collection.find({"_id": {"$in": matched_ids}}))
                    
                    # Sort them by score
                    id_to_place = {place["_id"]: place for place in semantic_places}
                    semantic_places = [id_to_place[match_id] for match_id, _ in semantic_matches 
                                      if match_id in id_to_place]
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
    
    # If we don't have enough candidates, add some random places
    if len(candidate_places) < size:
        additional_places = list(
            places_collection.find({"_id": {"$nin": list(added_ids)}})
            .limit(size - len(candidate_places))
        )
        candidate_places.extend(additional_places)
    
    return candidate_places

def rank_places(places, user_id):
    """
    Rank places based on similarity to user preferences.
    Returns a list of places sorted by relevance score.
    """
    user_prefs = get_user_preferences(user_id)
    if not user_prefs:
        # If no preferences, sort by rating
        return sorted(places, key=lambda x: float(x.get("rating", 0)), reverse=True)
    
    preferred_categories = user_prefs.get("preferred_categories", [])
    preferred_tags = user_prefs.get("preferred_tags", [])
    
    # Calculate scores for each place
    scored_places = []
    for place in places:
        score = 0
        
        # Category match (0-100 points)
        if place.get("category") in preferred_categories:
            score += 60  # Heavy weight on category match
        
        # Tags match (0-40 points per tag, up to 40)
        tag_score = 0
        for tag in place.get("tags", []):
            if tag in preferred_tags:
                tag_score += 40 / len(preferred_tags) if preferred_tags else 0
        score += min(tag_score, 40)
        
        # Rating boost (0-40 points)
        # Convert rating to float, default to 0 if missing or invalid
        try:
            rating = float(place.get("rating", 0))
            score += rating * (40 / 5)  # Scale rating (0-5) to (0-40)
        except (ValueError, TypeError):
            pass
        
        # Likes boost (0-30 points)
        likes = place.get("likes", 0)
        if likes > 0:
            # Apply diminishing returns for likes
            score += min(30, likes * 3)
            
        # Interactions boost (0-30 points)
        # Check if user has interacted with this place (views, saves)
        interactions = list(interactions_collection.find({
            "user_id": user_id,
            "place_id": place.get("_id", "")
        }).limit(1))
        
        if interactions:
            score += 30  # Boost places the user has interacted with
            
        scored_places.append((place, score))
    
    # Sort by score (descending) and return just the places
    return [p[0] for p in sorted(scored_places, key=lambda x: x[1], reverse=True)]

def filter_shown_places(places, user_id, include_from_last=5):
    """
    Filter out places that have been shown to the user recently,
    but include some from the most recent request for continuity.
    """
    # Get places shown to this user, ordered by descending timestamp
    shown_records = list(shown_places_collection.find(
        {"user_id": user_id}
    ).sort("timestamp", -1))
    
    # Get recently shown place IDs (all except those from the most recent request)
    recent_place_ids = set()
    most_recent_place_ids = set()
    
    for i, record in enumerate(shown_records):
        shown_id = record.get("place_id")
        if i < include_from_last:
            most_recent_place_ids.add(shown_id)
        else:
            recent_place_ids.add(shown_id)
    
    # Filter out places that were shown recently (except for the most recent ones)
    filtered_places = []
    recent_places = []
    
    for place in places:
        place_id = place.get("_id")
        if place_id in most_recent_place_ids:
            recent_places.append(place)
        elif place_id not in recent_place_ids:
            filtered_places.append(place)
    
    # If we don't have enough places after filtering, add some back from the shown places
    if len(filtered_places) < 10:
        # Return all we have
        return filtered_places + recent_places
    
    # Add some places from the most recent request (for continuity)
    # Take top places from filtered, reserve space for recent places
    return filtered_places[:(10-len(recent_places))] + recent_places

def get_trending_places(size=3):
    """Get trending places based on recent interactions."""
    # Get places with most interactions in the last 7 days
    one_week_ago = datetime.now() - timedelta(days=7)
    
    # Aggregate to find most interacted places
    try:
        pipeline = [
            # Match recent interactions
            {"$match": {"timestamp": {"$gte": one_week_ago}}},
            # Group by place_id and count interactions
            {"$group": {"_id": "$place_id", "count": {"$sum": 1}}},
            # Sort by count descending
            {"$sort": {"count": -1}},
            # Limit to specified size
            {"$limit": size}
        ]
        
        trending_ids = [doc["_id"] for doc in interactions_collection.aggregate(pipeline)]
        
        # Get the full place documents
        trending_places = []
        for place_id in trending_ids:
            place = places_collection.find_one({"_id": place_id})
            if place:
                trending_places.append(place)
                
        return trending_places
    except Exception as e:
        logger.error(f"Error getting trending places: {str(e)}")
        # Fallback to top-rated places if aggregation fails
        return list(places_collection.find().sort("rating", -1).limit(size))

def get_discovery_places(user_id, size=2):
    """Get discovery places (random places from categories the user hasn't explored)."""
    # Get user's interests
    user_prefs = get_user_preferences(user_id)
    if not user_prefs:
        # No preferences - return random places
        return list(places_collection.aggregate([{"$sample": {"size": size}}]))
    
    # Get categories the user has shown interest in
    preferred_categories = set(user_prefs.get("preferred_categories", []))
    
    # Get all distinct categories
    all_categories = places_collection.distinct("category")
    
    # Find categories the user hasn't explored
    unexplored_categories = [c for c in all_categories if c not in preferred_categories]
    
    # If user has explored all categories, pick random ones
    if not unexplored_categories:
        unexplored_categories = all_categories
    
    # Select random categories for discovery
    discovery_categories = random.sample(
        unexplored_categories, 
        min(len(unexplored_categories), 2)
    )
    
    # Get places from these categories
    discovery_places = []
    places_needed = size
    
    for category in discovery_categories:
        if places_needed <= 0:
            break
            
        # Get random places from this category
        category_places = list(places_collection.find(
            {"category": category}
        ).limit(places_needed))
        
        discovery_places.extend(category_places)
        places_needed -= len(category_places)
    
    # If we still need more places, get random ones
    if places_needed > 0:
        existing_ids = [p["_id"] for p in discovery_places]
        random_places = list(places_collection.find(
            {"_id": {"$nin": existing_ids}}
        ).limit(places_needed))
        
        discovery_places.extend(random_places)
    
    return discovery_places[:size]  # Ensure we don't exceed requested size

def track_shown_places(user_id, places):
    """Track places that have been shown to the user."""
    if not places:
        return
        
    # Get current timestamp
    now = datetime.now()
    
    # Prepare bulk insert documents
    documents = []
    for place in places:
        documents.append({
            "user_id": user_id,
            "place_id": place.get("_id"),
            "timestamp": now,
            "expires_at": now + timedelta(hours=6)  # Expire after 6 hours
        })
    
    # Insert records (if any)
    if documents:
        try:
            shown_places_collection.insert_many(documents)
        except Exception as e:
            logger.error(f"Error tracking shown places: {str(e)}")

def generate_final_recommendations(user_id, num_recommendations=10):
    """
    Generate the final recommendations using the multi-stage process:
    1. Get candidate places based on user preferences
    2. Rank them by relevance
    3. Filter out recently shown places
    4. Add trending and discovery places
    5. Track shown places
    """
    logger.info(f"Generating recommendations for user {user_id}")
    
    # 1. Get candidate places
    candidates = get_candidate_places(user_id, size=30)
    logger.info(f"Found {len(candidates)} candidate places")
    
    # 2. Rank candidates by relevance
    ranked_places = rank_places(candidates, user_id)
    logger.info(f"Ranked {len(ranked_places)} places")
    
    # 3. Filter out places the user has already seen
    filtered_places = filter_shown_places(ranked_places, user_id)
    logger.info(f"After filtering shown places: {len(filtered_places)} places")
    
    # 4. Add trending and discovery places
    final_count = min(len(filtered_places), num_recommendations - 5)  # Reserve 5 slots
    final_places = filtered_places[:final_count]
    
    # Add trending places (if we have space)
    trending_places = get_trending_places()
    # Only add trending places not already in the list
    existing_ids = {p["_id"] for p in final_places}
    for place in trending_places:
        if len(final_places) >= num_recommendations:
            break
        if place["_id"] not in existing_ids:
            final_places.append(place)
            existing_ids.add(place["_id"])
    
    # Add discovery places (if we have space)
    discovery_places = get_discovery_places(user_id)
    for place in discovery_places:
        if len(final_places) >= num_recommendations:
            break
        if place["_id"] not in existing_ids:
            final_places.append(place)
            existing_ids.add(place["_id"])
    
    # 5. Track shown places
    track_shown_places(user_id, final_places)
    
    # Return final recommendations
    return final_places[:num_recommendations]

async def background_cache_recommendations(user_id, num_entries=6):
    """
    Background task to generate and cache multiple recommendation sets.
    """
    logger.info(f"Starting background caching for user {user_id}, generating {num_entries} entries")
    
    # Use a semaphore-like approach to prevent multiple processes for the same user
    cache_lock_key = f"cache_lock_{user_id}"
    
    # Check if already being generated
    lock = recommendations_cache_collection.find_one({"_id": cache_lock_key})
    
    if lock:
        # Check if the lock is stale (older than 5 minutes)
        lock_time = lock.get("timestamp", datetime.min)
        if isinstance(lock_time, str):
            try:
                lock_time = datetime.fromisoformat(lock_time.replace('Z', '+00:00'))
            except Exception:
                lock_time = datetime.min
                
        if (datetime.now() - lock_time).total_seconds() < 300:  # 5 minutes
            logger.info(f"Cache generation already in progress for user {user_id}")
            return
            
        logger.info(f"Found stale lock for user {user_id}, proceeding with cache generation")
    
    # Set lock
    recommendations_cache_collection.update_one(
        {"_id": cache_lock_key},
        {"$set": {"timestamp": datetime.now()}},
        upsert=True
    )
    
    try:
        # Get highest sequence number
        highest_seq = recommendations_cache_collection.find_one(
            {"user_id": user_id, "_id": {"$ne": cache_lock_key}},
            sort=[("sequence", -1)]
        )
        
        next_seq = (highest_seq.get("sequence", -1) + 1) if highest_seq else 0
        
        # Generate and store recommendations
        for i in range(num_entries):
            # Generate a new set of recommendations
            recommendations = generate_final_recommendations(user_id)
            
            # Store in cache with sequence number
            recommendations_cache_collection.insert_one({
                "user_id": user_id,
                "sequence": next_seq + i,
                "recommendations": recommendations,
                "timestamp": datetime.now()
            })
            
            logger.info(f"Cached recommendations for user {user_id}, sequence {next_seq + i}")
            
            # Short sleep to prevent MongoDB overload
            await asyncio.sleep(0.5)
            
        logger.info(f"Successfully cached {num_entries} recommendation sets for user {user_id}")
        
    except Exception as e:
        logger.error(f"Error in background caching for user {user_id}: {str(e)}")
        
    finally:
        # Remove lock
        recommendations_cache_collection.delete_one({"_id": cache_lock_key})

# --- PART 4: ROADMAP ALGORITHM FUNCTIONS ---

# --- Roadmap System: Filtering Functions ---

def get_filtered_places(user_id):
    """
    Get places filtered by user's travel preferences including budget.
    
    Args:
        user_id: The user ID to retrieve filtered places for.
        
    Returns:
        List of places matching the user's travel preferences.
    """
    # Fetch user preferences (including budget)
    travel_prefs = get_user_travel_preferences(user_id)
    general_prefs = get_user_preferences(user_id)  # Get general preferences including budget
    
    if not travel_prefs or not travel_prefs.get("destinations", []):
        return list(places_collection.find())  # Return all places if no preferences are found
    
    # Get all places initially
    all_places = list(places_collection.find())
    
    # Step 1: Filter by preferred cities (location filter) - Apply this FIRST
    preferred_cities = travel_prefs["destinations"]
    filtered_places = []
    
    for place in all_places:
        location = place.get("location", {})
        city = location.get("city", "")
        if city in preferred_cities:
            filtered_places.append(place)
    
    # If no places match preferred cities, use all places
    if not filtered_places:
        logger.warning(f"No places found in preferred cities, using all places")
        filtered_places = all_places
    
    # Step 2: Filter by appropriate travel time (seasonal filter)
    travel_dates = travel_prefs.get("travel_dates", "")
    if travel_dates:
        target_month = parse_travel_dates(travel_dates)
        logger.info(f"Filtering places for month: {target_month}")
        
        if target_month:
            time_filtered_places = [
                place for place in filtered_places
                if "appropriate_time" in place and target_month in place["appropriate_time"]
            ]
            
            # Only use time-filtered places if we have results
            if time_filtered_places:
                logger.info(f"Found {len(time_filtered_places)} places appropriate for {target_month}")
                filtered_places = time_filtered_places
            else:
                logger.warning(f"No places found for month {target_month}, keeping previous filter results")
    
    # Step 3: Filter by budget
    user_budget = travel_prefs.get("budget", "medium")  # Get budget from travel_prefs instead
    budget_filtered_places = [
        place for place in filtered_places
        if "budget" in place and user_budget in place["budget"]
    ]
    
    # Only use budget-filtered places if we have results
    if budget_filtered_places:
        logger.info(f"Found {len(budget_filtered_places)} places matching budget: {user_budget}")
        filtered_places = budget_filtered_places
    else:
        logger.warning(f"No places found matching budget {user_budget}, keeping previous filter results")
    
    # Step 4: Filter by accessibility needs
    required_accessibility = travel_prefs.get("accessibility_needs", [])
    if required_accessibility:
        accessibility_filtered_places = [
            place for place in filtered_places
            if any(need in place.get("accessibility", []) for need in required_accessibility)
        ]
        
        # Only use accessibility-filtered places if we have results
        if accessibility_filtered_places:
            logger.info(f"Found {len(accessibility_filtered_places)} places matching accessibility needs")
            filtered_places = accessibility_filtered_places
        else:
            logger.warning(f"No places found matching accessibility needs, keeping previous filter results")
    
    # Step 5: Filter by group_type (30% weight in recommendations)
    group_type = travel_prefs.get("group_type", "")
    logger.info(f"User group_type: {group_type}")  # Log the user's group type
    
    if group_type:
        # Now filter by group_type (not suitable_for)
        group_filtered_places = [
            place for place in filtered_places
            if "group_type" in place and (
                (isinstance(place["group_type"], list) and group_type in place["group_type"]) or
                (not isinstance(place["group_type"], list) and group_type == place["group_type"])
            )
        ]
        
        logger.info(f"Found {len(group_filtered_places)} places matching group type: {group_type}")
        
        # If we have group-filtered places, apply 30% weighting
        if group_filtered_places:
            original_count = int(len(filtered_places) * 0.7)
            group_count = len(filtered_places) - original_count
            
            # Keep top 70% of original filtered places
            final_places = filtered_places[:original_count]
            
            # Add unique group-filtered places (up to 30%)
            existing_ids = {place["_id"] for place in final_places}
            for place in group_filtered_places:
                if place["_id"] not in existing_ids and len(final_places) < len(filtered_places):
                    final_places.append(place)
                    existing_ids.add(place["_id"])
                    
                    # Stop if we've added enough places
                    if len(final_places) >= (original_count + group_count):
                        break
            
            filtered_places = final_places
            logger.info(f"Applied group type filtering with 30% weight for {group_type}")
        else:
            logger.warning(f"No places found matching group type {group_type}, keeping previous filter results")
    else:
        logger.warning("No group type specified for user, skipping group type filtering")
    
    # If we have no places after all filters, return all places (fallback)
    if not filtered_places:
        logger.warning("No places left after all filters, returning all places")
        return all_places
    
    return filtered_places

def get_critical_filtered_places(user_id):
    """
    Filter places by critical constraints only (location).
    
    Args:
        user_id: The user ID to retrieve preferences for
        
    Returns:
        List of places in the user's preferred destinations
    """
    # Get travel preferences
    travel_prefs = get_user_travel_preferences(user_id)
    
    # If no preferences, return all places
    if not travel_prefs or not travel_prefs.get("destinations"):
        logger.warning(f"No destination preferences found for user {user_id}, using all places")
        return list(places_collection.find())
    
    # Get all places
    all_places = list(places_collection.find())
    
    # Critical filter: Preferred cities (location filter)
    preferred_cities = travel_prefs.get("destinations", [])
    filtered_places = []
    
    for place in all_places:
        location = place.get("location", {})
        city = location.get("city", "")
        if city in preferred_cities:
            filtered_places.append(place)
    
    # If no places match preferred cities, use all places as fallback
    if not filtered_places:
        logger.warning(f"No places found in preferred cities, using all places")
        return all_places
    
    logger.info(f"Critical filtering found {len(filtered_places)} places in preferred destinations")
    return filtered_places

# --- Roadmap System: Ranking Functions ---

def compute_similarity(place, user_prefs, travel_prefs=None):
    """Compute similarity score between place and user preferences"""
    score = 0
    
    # Fixed to handle potential missing fields or different structure
    place_category = place.get("category", "")
    place_tags = place.get("tags", [])
    place_accessibility = place.get("accessibility", [])
    
    # Check if place category is in user's preferred categories
    if user_prefs.get("preferred_categories") and place_category in user_prefs["preferred_categories"]:
        score += 5
        
    # Check if any place tags are in user's preferred tags
    if user_prefs.get("preferred_tags"):
        for tag in place_tags:
            if tag in user_prefs["preferred_tags"]:
                score += 3
                
    # Check if place meets any accessibility needs (if travel_prefs is provided)
    if travel_prefs and travel_prefs.get("accessibility_needs"):
        for need in travel_prefs["accessibility_needs"]:
            if need in place_accessibility:
                score += 2
                
    return score

def rank_places_content_based_hybrid(user_id, filtered_places):
    """
    Rank pre-filtered places by content-based similarity to user preferences.
    
    Args:
        user_id: User ID to get preferences for
        filtered_places: List of pre-filtered place documents
        
    Returns:
        List of places sorted by similarity to user preferences
    """
    user_prefs = get_user_preferences(user_id)
    travel_prefs = get_user_travel_preferences(user_id)
    
    if not user_prefs or not travel_prefs:
        logger.warning(f"No preferences found for user {user_id}, ranking by rating")
        return sorted(filtered_places, key=lambda x: float(x.get("rating", 0)), reverse=True)
        
    # Sort places by similarity score
    ranked_places = sorted(
        filtered_places,
        key=lambda p: compute_similarity(p, user_prefs, travel_prefs),
        reverse=True
    )
    
    return ranked_places

def get_collaborative_scores_hybrid(user_id, filtered_places):
    """
    Get collaborative filtering scores for pre-filtered places.
    
    Args:
        user_id: User ID to get similar users for
        filtered_places: List of pre-filtered place documents
        
    Returns:
        Dictionary mapping place IDs to collaborative scores
    """
    user_prefs = get_user_preferences(user_id)
    if not user_prefs:
        return {}
        
    # Find users with similar preferences
    similar_users = list(users_collection.find({
        "$or": [
            {"preferences.categories": {"$in": user_prefs.get("preferred_categories", [])}},
            {"preferences.tags": {"$in": user_prefs.get("preferred_tags", [])}}
        ]
    }))
    
    # Create a set of filtered place IDs for quick lookup
    filtered_place_ids = {place.get("_id") for place in filtered_places}
    
    # Dictionary to store place scores
    place_scores = {}
    
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
    
    for user in similar_users:
        # Skip the current user
        if user.get("_id") == user_id:
            continue
            
        # Get interactions for this user
        interactions = list(interactions_collection.find({"user_id": user.get("_id")}))
        
        for interaction in interactions:
            place_id = interaction.get("place_id")
            
            # Skip if not in our filtered places
            if place_id not in filtered_place_ids:
                continue
                
            # Get interaction type/action
            action = interaction.get("action", interaction.get("interaction_type", ""))
            
            # Get timestamp (fallback to current time if missing)
            timestamp = interaction.get("timestamp", datetime.now())
            
            # Calculate weighted score with time decay
            weight = action_weights.get(action, 0)
            weighted_score = apply_time_decay(weight, timestamp)
            
            # Update place scores
            if place_id in place_scores:
                place_scores[place_id] += weighted_score
            else:
                place_scores[place_id] = weighted_score
                
    return place_scores

def get_backup_places(user_id, existing_places):
    """Get backup places in case we don't have enough recommendations"""
    travel_prefs = get_user_travel_preferences(user_id)
    if not travel_prefs or not travel_prefs.get("destinations", []):
        return []
        
    preferred_cities = travel_prefs.get("destinations", [])
    
    # Get places not already in existing_places
    existing_ids = [p["_id"] for p in existing_places]
    all_places = list(places_collection.find({"_id": {"$nin": existing_ids}}))
    
    selected_places = list(places_collection.find({"location.city": {"$in": preferred_cities}}))
    if not selected_places:
        return []
        
    # Handle nested numeric values for coordinates
    def get_coordinate(location_obj, key):
        coordinate = location_obj.get(key, 0)
        # Handle if coordinate is a MongoDB NumberDouble object
        if isinstance(coordinate, dict) and "$numberDouble" in coordinate:
            return float(coordinate["$numberDouble"])
        return float(coordinate)
        
    # Calculate average location from preferred places
    total_places = len(selected_places)
    sum_lat = sum(get_coordinate(p["location"], "latitude") for p in selected_places)
    sum_lon = sum(get_coordinate(p["location"], "longitude") for p in selected_places)
    
    avg_lat = sum_lat / total_places if total_places > 0 else 0
    avg_lon = sum_lon / total_places if total_places > 0 else 0
    
    user_center_location = (avg_lat, avg_lon)
    
    # Sort places by distance to center
    def get_distance(place):
        try:
            place_lat = get_coordinate(place["location"], "latitude")
            place_lon = get_coordinate(place["location"], "longitude")
            return geodesic(user_center_location, (place_lat, place_lon)).km
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return float('inf')  # Return infinite distance on error
            
    all_places.sort(key=get_distance)
    
    return all_places[:10]

# --- PART 5: ROADMAP GENERATION AND FORMATTING ---

def generate_hybrid_roadmap(user_id):
    """
    Generate the final roadmap using the hybrid approach:
    1. Apply critical filters first (location)
    2. Run recommendations on this subset
    3. Apply softer filters with weights
    
    Args:
        user_id: User ID to generate roadmap for
        
    Returns:
        List of recommended places
    """
    logger.info(f"🎯 Generating Hybrid Roadmap for {user_id}...")
    
    # Get user preferences for logging
    user_prefs = get_user_preferences(user_id)
    travel_prefs = get_user_travel_preferences(user_id)
    
    logger.info(f"User preferences: {user_prefs}")
    logger.info(f"Travel preferences: {travel_prefs}")
    
    # 1. CRITICAL FILTERS FIRST: Apply absolute essentials (location)
    critical_filtered_places = get_critical_filtered_places(user_id)
    logger.info(f"Found {len(critical_filtered_places)} places after critical filtering")
    
    # 2. RUN RECOMMENDATIONS on pre-filtered subset
    
    # Content-based filtering
    content_based_places = rank_places_content_based_hybrid(user_id, critical_filtered_places)
    logger.info(f"Content-based ranking complete: {len(content_based_places)} places")
    
    # Collaborative filtering scores
    try:
        collaborative_scores = get_collaborative_scores_hybrid(user_id, critical_filtered_places)
        logger.info(f"Collaborative scores generated for {len(collaborative_scores)} places")
    except Exception as e:
        logger.error(f"Error in collaborative filtering: {str(e)}")
        collaborative_scores = {}
    
    # Combine scores with weights (70% content-based, 30% collaborative)
    final_scored_places = []
    
    for i, place in enumerate(content_based_places):
        # Content-based score based on position (higher = better)
        content_score = 1.0 - (i / len(content_based_places)) if content_based_places else 0
        
        # Get collaborative score if available
        collab_score = collaborative_scores.get(place.get("_id", ""), 0)
        
        # Normalize collaborative score (if we have scores)
        if collaborative_scores:
            max_collab = max(collaborative_scores.values()) if collaborative_scores.values() else 1
            if max_collab > 0:
                collab_score = collab_score / max_collab
        
        # Combined weighted score
        final_score = (content_score * 0.7) + (collab_score * 0.3)
        final_scored_places.append((place, final_score))
    
    # Sort by final score
    sorted_places = [p[0] for p in sorted(final_scored_places, key=lambda x: x[1], reverse=True)]
    
    # 3. APPLY SOFTER FILTERS WITH WEIGHTS
    final_places = []
    
    if travel_prefs:
        # Get the filters we want to apply with weights
        budget = travel_prefs.get("budget", "medium")
        required_accessibility = travel_prefs.get("accessibility_needs", [])
        group_type = travel_prefs.get("group_type", "")
        travel_dates = travel_prefs.get("travel_dates", "")
        target_month = parse_travel_dates(travel_dates) if travel_dates else None
        
        # Scoring function for soft constraints
        def soft_constraint_score(place):
            score = 0.0
            
            # Budget match (30% weight)
            if "budget" in place and budget in place["budget"]:
                score += 0.3
            
            # Accessibility match (20% weight)
            if required_accessibility:
                if any(need in place.get("accessibility", []) for need in required_accessibility):
                    score += 0.2
            else:
                # If no accessibility needs specified, give full points
                score += 0.2
            
            # Group type match (30% weight)
            if group_type and "group_type" in place:
                if (isinstance(place["group_type"], list) and group_type in place["group_type"]) or \
                   (not isinstance(place["group_type"], list) and group_type == place["group_type"]):
                    score += 0.3
            else:
                # If no group type specified, give full points
                score += 0.3
            
            # Time/seasonal match (20% weight)
            if target_month and "appropriate_time" in place and target_month in place["appropriate_time"]:
                score += 0.2
            elif not target_month:
                # If no time specified, give full points
                score += 0.2
            
            return score
        
        # Apply soft constraints with weighted scoring
        soft_filtered_places = []
        for place in sorted_places:
            soft_score = soft_constraint_score(place)
            soft_filtered_places.append((place, soft_score))
        
        # Sort by descending soft score, giving preference to places that meet more soft constraints
        final_places = [p[0] for p in sorted(soft_filtered_places, key=lambda x: x[1], reverse=True)]
    else:
        # If no travel preferences, just use the ranked places
        final_places = sorted_places
    
    # Add saved places from the user if they're in the filtered cities
    user = users_collection.find_one({"_id": user_id})
    if user and "saved_places" in user:
        saved_place_ids = user.get("saved_places", [])
        logger.info(f"User has {len(saved_place_ids)} saved places")
        
        for saved_place_id in saved_place_ids:
            # Skip if we already have this place in final_places
            if any(p.get("_id") == saved_place_id for p in final_places):
                continue
            
            # Get saved place details
            saved_place = places_collection.find_one({"_id": saved_place_id})
            if saved_place:
                # Check if it's in a preferred city
                if travel_prefs and "destinations" in travel_prefs:
                    location = saved_place.get("location", {})
                    city = location.get("city", "")
                    if city in travel_prefs["destinations"]:
                        # Add to the beginning of the list
                        final_places.insert(0, saved_place)
                        logger.info(f"Added saved place {saved_place.get('name')} to recommendations")
    
    # Ensure we have at most 10 places
    final_roadmap = final_places[:10]
    logger.info(f"Final hybrid roadmap has {len(final_roadmap)} places")
    
    # Fallback to popular places if we have no recommendations
    if not final_roadmap:
        logger.warning(f"No places found after hybrid filtering for user {user_id}, returning popular places")
        try:
            # Get places with highest ratings
            popular_places = list(places_collection.find().sort([("average_rating", -1)]).limit(10))
            if not popular_places:
                popular_places = list(places_collection.find().sort([("rating", -1)]).limit(10))
            return popular_places
        except Exception as e:
            logger.error(f"Error getting popular places: {str(e)}")
    
    return final_roadmap

def simplify_roadmap_to_list(roadmap_data):
    """
    Convert roadmap data to a simple list format without design elements
    """
    if not roadmap_data:
        return []
        
    simplified_list = []
    
    # Process a list of roadmaps
    if isinstance(roadmap_data, list):
        for item in roadmap_data:
            if isinstance(item, dict):
                # Safely extract values with fallbacks for MongoDB document structure
                def get_nested_value(obj, key, default=""):
                    if "." in key:
                        parts = key.split(".", 1)
                        if parts[0] in obj and isinstance(obj[parts[0]], dict):
                            return get_nested_value(obj[parts[0]], parts[1], default)
                        return default
                    return obj.get(key, default)
                    
                # Handle numeric values that might be MongoDB objects
                def get_numeric_value(obj, key, default=0.0):
                    value = get_nested_value(obj, key, default)
                    if isinstance(value, dict) and "$numberDouble" in value:
                        return float(value["$numberDouble"])
                    if isinstance(value, dict) and "$numberInt" in value:
                        return int(value["$numberInt"])
                    return value
                    
                # Extract location data safely
                location = item.get("location", {})
                
                # Extract essential information for each place
                place_info = {
                    "id": str(item.get("_id", "")),
                    "name": item.get("name", ""),
                    "description": item.get("description", ""),
                    "category": item.get("category", ""),
                    "location": {
                        "city": location.get("city", ""),
                        "country": location.get("country", ""),
                        "latitude": get_numeric_value(location, "latitude", 0.0),
                        "longitude": get_numeric_value(location, "longitude", 0.0)
                    },
                    "rating": get_numeric_value(item, "average_rating", 0),  # Use average_rating instead of rating
                    "tags": item.get("tags", []),
                    "accessibility": item.get("accessibility", []),
                }
                simplified_list.append(place_info)
            elif isinstance(item, str):
                simplified_list.append(item)
                
    return simplified_list

async def get_roadmap_with_caching(user_id: str):
    """
    Get a roadmap for a user, with caching
    
    1. Check if user's travel preferences have changed since last generation
    2. If unchanged, return cached roadmap if available
    3. If changed or no cache, generate a new roadmap
    """
    # Get user's travel preferences
    travel_prefs = get_user_travel_preferences(user_id)
    
    # Find existing roadmap for this user
    existing_roadmap = roadmaps_collection.find_one({"user_id": user_id})
    
    # If we have an existing roadmap, check if preferences have changed
    if existing_roadmap:
        cached_prefs = existing_roadmap.get("travel_preferences", {})
        cached_prefs_hash = hash(str(cached_prefs))
        current_prefs_hash = hash(str(travel_prefs)) if travel_prefs else 0
        
        # If preferences haven't changed, use cached roadmap
        if cached_prefs_hash == current_prefs_hash:
            logger.info(f"Using cached roadmap for user {user_id} - preferences unchanged")
            return existing_roadmap["roadmap_data"]
        else:
            logger.info(f"Travel preferences changed for user {user_id}, generating new roadmap")
    
    # Generate new roadmap
    logger.info(f"Generating new roadmap for user {user_id}")
    roadmap_data = generate_hybrid_roadmap(user_id)
    
    # Store in cache
    now = datetime.now()
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

# --- PART 6: API ENDPOINTS AND SERVER STARTUP ---

# --- Root endpoint ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Travel API", "status": "active"}

# --- Recommendations API Endpoints ---

@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: str, background_tasks: BackgroundTasks):
    """
    Get recommendations for a specific user.
    
    If cached recommendations exist, return and replenish cache.
    Otherwise, generate recommendations on-demand.
    """
    try:
        # Try to get a cached entry first
        cached_entry = recommendations_cache_collection.find_one(
            {"user_id": user_id},
            sort=[("sequence", 1)]  # Get the oldest entry (lowest sequence)
        )
        
        if cached_entry:
            # We have a cached entry - return it and replenish cache if running low
            recommendations = cached_entry.get("recommendations", [])
            
            # Delete the used cache entry
            recommendations_cache_collection.delete_one({"_id": cached_entry["_id"]})
            
            # Count remaining cache entries
            remaining_count = recommendations_cache_collection.count_documents({"user_id": user_id})
            
            # If we're running low on cache entries (≤3), schedule background replenishment
            if remaining_count <= 3:
                logger.info(f"Cache running low for user {user_id}, scheduling replenishment")
                background_tasks.add_task(background_cache_recommendations, user_id, 6)
                
            logger.info(f"Returning cached recommendations for user {user_id}, seq: {cached_entry.get('sequence')}")
            
            # Return recommendations from cache
            return {
                "success": True,
                "user_id": user_id,
                "cached": True,
                "recommendations": recommendations
            }
        else:
            # No cache - generate on demand
            logger.info(f"No cached recommendations for user {user_id}, generating on demand")
            recommendations = generate_final_recommendations(user_id)
            
            # Store as sequence 0 (on-demand generation)
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
                "cached": False,
                "recommendations": recommendations
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
    lock = recommendations_cache_collection.find_one({"_id": cache_lock_key})
    
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
        
        # Search in name and description fields
        all_places = list(places_collection.find())
        results = []
        
        for place in all_places:
            score = 0
            
            # Exact name match - highest score
            if query.lower() in place.get("name", "").lower():
                score = 1.0
            # Description match - partial score
            elif "description" in place and query.lower() in place.get("description", "").lower():
                score = 0.5
                
            if score > 0:
                # Add to results with score
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
            "results": final_results
        }
        
    except Exception as e:
        logger.error(f"Error searching places: {str(e)}")
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
        
        return {"success": True, "user_id": user_id, "data": simplified_list}
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
        
        return {"success": True, "user_id": user_id, "data": simplified_list}
    except Exception as e:
        logger.error(f"Error generating roadmap: {str(e)}")
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
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=port)
    
