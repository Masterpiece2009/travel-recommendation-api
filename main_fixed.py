import os
import datetime
import asyncio
import math
import random
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
import pymongo
import spacy
import nest_asyncio
import uvicorn
from sklearn.preprocessing import MinMaxScaler

# Enable nested asyncio for background tasks
nest_asyncio.apply()

# Initialize FastAPI
app = FastAPI(title="Travel Recommendation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to MongoDB
mongo_uri = os.environ.get("MONGO_URI")
mongo_password = os.environ.get("MONGO_PASSWORD")

# Replace password placeholder with actual password
if mongo_uri and mongo_password:
    mongo_uri = mongo_uri.replace("<db_password>", mongo_password)
    client = pymongo.MongoClient(mongo_uri)
    # Use correct database name
    db = client.travel_app
else:
    raise ValueError("MongoDB URI or password not provided in environment variables")

# TTL in seconds (6 hours)
TTL_SECONDS = 6 * 60 * 60

# ✅ IMPROVED: Initialize all collections with proper TTL indexes
def initialize_collections():
    """Set up all collections with proper indexes and TTL configurations"""
    try:
        print("Initializing collections and indexes...")
        
        # 1. Recommendation Cache Collection
        cache_collection = db.recommendation_cache
        # Create index for efficient lookups
        cache_collection.create_index([
            ("user_id", pymongo.ASCENDING)
        ])
        
        # Create index for cache expiration (6 hours TTL)
        cache_collection.create_index([
            ("timestamp", pymongo.ASCENDING)
        ], expireAfterSeconds=TTL_SECONDS)
        
        # 2. Cache Locks Collection
        locks_collection = db.cache_locks
        # Create index for lock expiration (10 minutes TTL)
        locks_collection.create_index([
            ("timestamp", pymongo.ASCENDING)
        ], expireAfterSeconds=600)
        
        # 3. User Shown Places Collection
        shown_places_collection = db.user_shown_places
        # Create index for user lookup
        shown_places_collection.create_index([
            ("user_id", pymongo.ASCENDING)
        ])
        
        # Create index for TTL expiration (6 hours, same as cache)
        shown_places_collection.create_index([
            ("timestamp", pymongo.ASCENDING)
        ], expireAfterSeconds=TTL_SECONDS)
        
        print("Collections initialized successfully with TTL indexes")
        return True
    except Exception as e:
        print(f"Error initializing collections: {e}")
        return False

# Initialize collections on startup
initialize_collections()

# ✅ Load NLP Model
try:
    # Load smaller model suitable for Railway deployment
    nlp = spacy.load("en_core_web_sm")
    print("NLP model loaded successfully")
except Exception as e:
    print(f"Error loading NLP model: {e}")
    # Create blank model as fallback
    nlp = spacy.blank("en")
    print("Using blank NLP model as fallback")

# ✅ Cache Management Functions
def get_user_cached_recommendations(user_id: str):
    """Get all cached recommendations for a user, sorted by sequence"""
    try:
        cache_entries = list(db.recommendation_cache.find(
            {"user_id": user_id}
        ).sort("sequence", 1))
        return cache_entries
    except Exception as e:
        print(f"Error getting user's cached recommendations: {e}")
        return []

def get_next_sequence(user_id: str):
    """Get the next sequence number for a user's cache entries"""
    try:
        highest_seq = db.recommendation_cache.find_one(
            {"user_id": user_id},
            sort=[("sequence", pymongo.DESCENDING)]
        )
        
        if highest_seq and "sequence" in highest_seq:
            return highest_seq["sequence"] + 1
        else:
            return 1  # Start from 1 if no entries exist
    except Exception as e:
        print(f"Error getting next sequence number: {e}")
        return 1  # Default to 1 on error

def count_remaining_cache(user_id: str):
    """Count how many cache entries remain for a user"""
    try:
        count = db.recommendation_cache.count_documents({"user_id": user_id})
        return count
    except Exception as e:
        print(f"Error counting cache entries: {e}")
        return 0

def store_cache_entry(user_id: str, recommendations: dict, sequence: int):
    """Store a new cache entry with sequence number"""
    try:
        # Add timestamp and sequence number
        cache_entry = {
            "user_id": user_id,
            "recommendations": recommendations,
            "sequence": sequence,
            "timestamp": datetime.now()
        }
        
        # Insert cache entry
        result = db.recommendation_cache.insert_one(cache_entry)
        print(f"Stored cache entry for user {user_id}, sequence {sequence}")
        return True
    except Exception as e:
        print(f"Error storing cache entry: {e}")
        return False

def clear_user_cache(user_id: str):
    """Clear all cache entries for a user"""
    try:
        result = db.recommendation_cache.delete_many({"user_id": user_id})
        print(f"Cleared {result.deleted_count} cache entries for user {user_id}")
        return result.deleted_count
    except Exception as e:
        print(f"Error clearing user cache: {e}")
        return 0

def acquire_cache_lock(user_id: str):
    """Try to acquire a lock for cache generation for a user"""
    try:
        lock_key = f"generating_{user_id}"
        existing_lock = db.cache_locks.find_one({"key": lock_key})
        if existing_lock:
            return False
        
        db.cache_locks.insert_one({
            "key": lock_key,
            "timestamp": datetime.now(),
            "user_id": user_id
        })
        return True
    except Exception as e:
        print(f"Error acquiring cache lock: {e}")
        return False

def release_cache_lock(user_id: str):
    """Release the cache generation lock for a user"""
    try:
        lock_key = f"generating_{user_id}"
        result = db.cache_locks.delete_one({"key": lock_key})
        return result.deleted_count > 0
    except Exception as e:
        print(f"Error releasing cache lock: {e}")
        return False

# ✅ User Data Functions
def get_user_data(user_id):
    """Get user data including preferences and history"""
    try:
        user = db.users.find_one({"_id": user_id})
        if not user:
            return None

        return {
            "preferences": user.get("preferences", {}),
            "saved_places": user.get("saved_places", []),
            "search_history": list(db.search_queries.find({"user_id": user_id})),
            "interactions": list(db.interactions.find({"user_id": user_id}))
        }
    except Exception as e:
        print(f"Error fetching user data: {e}")
        return None

# ✅ Places Management Functions
def get_previously_shown_places(user_id: str):
    """Get all places previously shown to a user"""
    try:
        user_shown = db.user_shown_places.find_one({"user_id": user_id})
        return user_shown.get("place_ids", []) if user_shown else []
    except Exception as e:
        print(f"Error getting previously shown places: {e}")
        return []

def get_last_shown_places(user_id: str):
    """Get only the places shown in the most recent request"""
    try:
        user_shown = db.user_shown_places.find_one({"user_id": user_id})
        return user_shown.get("last_shown_place_ids", []) if user_shown else []
    except Exception as e:
        print(f"Error getting last shown places: {e}")
        return []

def reset_user_shown_places(user_id: str):
    """Reset the tracking of places shown to a user"""
    try:
        result = db.user_shown_places.delete_one({"user_id": user_id})
        deleted = result.deleted_count > 0
        print(f"Reset shown places for user {user_id}, success: {deleted}")
        return deleted
    except Exception as e:
        print(f"Error resetting shown places: {e}")
        return False

def update_shown_places(user_id: str, new_place_ids, max_places=None):
    """Update the list of shown places for a user"""
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
        db.user_shown_places.update_one(
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
        print(f"Error updating shown places: {e}")
        return []

# ✅ Core Recommendation Algorithms

def get_candidate_places(user_prefs, user_id):
    """Get candidate places based on user preferences and search history"""
    try:
        categories = user_prefs.get("categories", [])

        # Handle empty categories
        if not categories:
            return []

        # Fetch places based on categories (60%)
        category_places = list(db.places.find({"category": {"$in": categories}}))

        # Fetch only the last 5 search queries for this user, sorted by timestamp
        search_queries = list(db.search_queries.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(5))

        # Extract keywords from search queries
        search_keywords = set()
        for query in search_queries:
            if "keywords" in query and isinstance(query["keywords"], list):
                for keyword in query["keywords"]:
                    search_keywords.add(keyword)

        # Get all places for semantic similarity matching
        all_places = list(db.places.find())

        # Initialize semantic matching scores
        keyword_place_scores = {}

        # Process each keyword with NLP
        for keyword in search_keywords:
            # Process the keyword with spaCy
            keyword_doc = nlp(keyword.lower())

            # Compare with tags for each place
            for place in all_places:
                place_id = place["_id"]

                # Initialize score for this place if it doesn't exist
                if place_id not in keyword_place_scores:
                    keyword_place_scores[place_id] = 0

                # Check each tag for similarity
                if "tags" in place and isinstance(place["tags"], list):
                    for tag in place["tags"]:
                        # Process the tag with spaCy
                        tag_doc = nlp(tag.lower())

                        # Calculate similarity score (0-1)
                        similarity = keyword_doc.similarity(tag_doc)

                        # Consider it a match if similarity is above threshold (0.6)
                        if similarity > 0.6:
                            # Add similarity score to place
                            keyword_place_scores[place_id] += similarity

        # Get top matching places based on semantic similarity
        semantic_matches = [(place_id, score) for place_id, score in keyword_place_scores.items() if score > 0]
        semantic_matches.sort(key=lambda x: x[1], reverse=True)

        # Get the top matches (limit to 30 for performance)
        top_semantic_matches = semantic_matches[:30]

        # Get the actual place documents for the matches
        keyword_places = []
        if top_semantic_matches:
            matched_ids = [match[0] for match in top_semantic_matches]
            keyword_places = list(db.places.find({"_id": {"$in": matched_ids}}))

            # Sort them in the same order as the semantic matches
            id_to_place = {place["_id"]: place for place in keyword_places}
            keyword_places = [id_to_place[match_id] for match_id, _ in top_semantic_matches if match_id in id_to_place]

        # Protect against empty results or division by zero
        total_category_places = len(category_places)
        category_count = min(total_category_places, max(1, int(total_category_places * 0.6))) if total_category_places > 0 else 0

        total_keyword_places = len(keyword_places)
        keyword_count = min(total_keyword_places, max(1, int(total_keyword_places * 0.4))) if total_keyword_places > 0 else 0

        # Combine results
        result = []
        if category_count > 0:
            result.extend(category_places[:category_count])
        if keyword_count > 0:
            result.extend(keyword_places[:keyword_count])

        return result
    except Exception as e:
        print(f"Error getting candidate places: {e}")
        return []

def get_collaborative_recommendations(user_id):
    """Get places from similar users based on various interactions"""
    try:
        user = db.users.find_one({"_id": user_id})
        if not user:
            return []

        # Get user preferences
        user_prefs = user.get("preferences", {})
        preferred_categories = user_prefs.get("categories", [])
        preferred_tags = user_prefs.get("tags", [])

        # Find similar users
        similar_users = list(db.users.find({
            "preferences.categories": {"$in": preferred_categories},
            "preferences.tags": {"$in": preferred_tags},
            "_id": {"$ne": user_id}
        }))

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

        # Track recommended places
        recommended_places = set()

        # Get existing interactions for user
        user_interactions = {}
        for i in db.interactions.find({"user_id": user_id}):
            if "place_id" in i and "interaction_type" in i:
                user_interactions[i["place_id"]] = i["interaction_type"]

        # Process interactions from similar users
        for similar_user in similar_users:
            interactions = list(db.interactions.find({"user_id": similar_user["_id"]}))

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
                weighted_score = apply_time_decay(weight, timestamp)

                # Only add positively scored places
                if weighted_score > 0:
                    recommended_places.add(place_id)

        return list(recommended_places)
    except Exception as e:
        print(f"Error in collaborative filtering: {e}")
        return []

def get_trending_places(limit=20):
    """Get places that are trending based on recent interactions"""
    try:
        # Get interactions from last 14 days
        two_weeks_ago = datetime.now().replace(tzinfo=None) - timedelta(days=14)

        # Find recent interactions and aggregate by place_id
        pipeline = [
            {
                "$match": {
                    "timestamp": {"$gte": two_weeks_ago},
                    "interaction_type": {"$in": ["like", "save", "share", "view"]}
                }
            },
            {
                "$group": {
                    "_id": "$place_id",
                    "interaction_count": {"$sum": 1}
                }
            },
            {"$sort": {"interaction_count": -1}},
            {"$limit": limit}
        ]

        trending_results = list(db.interactions.aggregate(pipeline))
        trending_place_ids = [result["_id"] for result in trending_results]

        # Get actual place documents
        trending_places = list(db.places.find({"_id": {"$in": trending_place_ids}}))

        # Sort by the original trending order
        id_to_place = {place["_id"]: place for place in trending_places}
        sorted_trending = [id_to_place[place_id] for place_id in trending_place_ids if place_id in id_to_place]

        return sorted_trending
    except Exception as e:
        print(f"Error getting trending places: {e}")
        return []

def get_discovery_places(user_id, limit=10):
    """Get places outside the user's normal patterns for discovery"""
    try:
        user_data = get_user_data(user_id)
        if not user_data:
            return []

        user_prefs = user_data["preferences"]
        user_categories = user_prefs.get("categories", [])

        # Get user's region preferences to exclude them
        user_regions = user_prefs.get("regions", [])

        # Find places in different categories but highly rated
        discovery_query = {
            "category": {"$nin": user_categories},
            "average_rating": {"$gte": 4.0}  # Only high-rated places
        }

        # If user has region preferences, include some places from other regions
        if user_regions:
            discovery_query["region"] = {"$nin": user_regions}

        # Get discovery places and sort by rating
        discovery_places = list(db.places.find(discovery_query).sort("average_rating", -1).limit(limit * 2))

        # If we don't have enough, try a broader search
        if len(discovery_places) < limit:
            fallback_places = list(db.places.find({
                "category": {"$nin": user_categories}
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

        return discovery_places[:limit]
    except Exception as e:
        print(f"Error getting discovery places: {e}")
        return []

def rank_places(candidate_places, user_id):
    """Rank places based on user engagement and popularity"""
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
            interactions_count = db.interactions.count_documents({"user_id": user_id, "place_id": place["_id"]})

            # Extract numeric values correctly
            avg_rating = extract_number(place.get("average_rating", 0))
            likes = extract_number(place.get("likes", 0))

            place["score"] = (
                0.4 * avg_rating +
                0.3 * likes / 10000 +
                0.3 * interactions_count
            )

        scores = [[p["score"]] for p in candidate_places]
        if scores:
            scaled_scores = scaler.fit_transform(scores)
            for i, place in enumerate(candidate_places):
                place["final_score"] = float(scaled_scores[i][0])  # Convert numpy type to float
        else:
            for place in candidate_places:
                place["final_score"] = 0  # Default score if no data available

        # Use final_score for sorting, not the objects themselves
        return sorted(candidate_places, key=lambda x: x.get("final_score", 0), reverse=True)
    except Exception as e:
        print(f"Error ranking places: {e}")
        # Use average_rating for fallback sorting, but extract numeric value first
        return sorted(candidate_places,
                     key=lambda x: extract_number(x.get("average_rating", 0)),
                     reverse=True)  # Fallback sorting

def refresh_shown_places(user_id, shown_place_ids, limit=10):
    """Re-rank and refresh previously shown places"""
    try:
        if not shown_place_ids:
            return []

        # Get the place documents
        shown_places = list(db.places.find({"_id": {"$in": shown_place_ids}}))

        if not shown_places:
            return []

        # Get recent interaction data for all users
        recent_date = datetime.now() - timedelta(days=7)

        # Count recent interactions for each place
        place_interaction_counts = {}
        for place_id in shown_place_ids:
            count = db.interactions.count_documents({
                "place_id": place_id,
                "timestamp": {"$gte": recent_date},
                "interaction_type": {"$in": ["like", "save", "share", "view"]}
            })
            place_interaction_counts[place_id] = count

        # Add recency score to places
        for place in shown_places:
            place["recency_score"] = place_interaction_counts.get(place["_id"], 0)

        # Sort by recency score and add some randomness
        refreshed_places = sorted(shown_places, key=lambda x: x.get("recency_score", 0) + random.random(), reverse=True)

        return refreshed_places[:limit]
    except Exception as e:
        print(f"Error refreshing shown places: {e}")
        return []

async def background_cache_recommendations(user_id: str, count: int = 6):
    """
    Background task to generate and cache multiple recommendation sequences
    Uses locking to prevent duplicate work
    """
    # Try to acquire the lock
    if not acquire_cache_lock(user_id):
        print(f"Cache generation already in progress for user {user_id}")
        return
    
    try:
        # Get the next sequence number
        start_sequence = get_next_sequence(user_id)
        print(f"Starting background caching for user {user_id}, starting from sequence {start_sequence}")
        
        # Get all previously shown places to maintain continuity
        all_shown_place_ids = get_previously_shown_places(user_id)
        
        # Generate and store each sequence
        for i in range(count):
            current_sequence = start_sequence + i
            
            # Generate new recommendations, building on previously shown places
            recommendations = generate_final_recommendations(user_id)
            
            # Store this set of recommendations in the cache
            if "recommendations" in recommendations:
                store_cache_entry(user_id, recommendations, current_sequence)
                
                # Update the list of shown places for next iteration
                if "recommendations" in recommendations:
                    new_place_ids = [p["_id"] for p in recommendations["recommendations"][:10]]
                    all_shown_place_ids = update_shown_places(user_id, new_place_ids, max_places=None)
            
            # Small delay to prevent overloading the server
            await asyncio.sleep(0.5)
            
        print(f"Completed background caching for user {user_id}")
    except Exception as e:
        print(f"Error in background caching: {e}")
    finally:
        # Always release the lock
        release_cache_lock(user_id)

def generate_final_recommendations(user_id):
    """Generate final recommendations by combining all algorithms"""
    try:
        user_data = get_user_data(user_id)
        if not user_data:
            return {"error": "User not found"}

        # Get previously shown places for this user
        previously_shown_place_ids = get_previously_shown_places(user_id)
        print(f"Found {len(previously_shown_place_ids)} previously shown places")

        # Get content-based and collaborative recommendations
        content_based = get_candidate_places(user_data["preferences"], user_id)
        print(f"Found {len(content_based)} content-based places")

        collaborative_place_ids = get_collaborative_recommendations(user_id)
        collaborative = list(db.places.find({"_id": {"$in": collaborative_place_ids}}))
        print(f"Found {len(collaborative)} collaborative places")

        # Combine and rank all places
        all_places = content_based + collaborative
        print(f"Total combined places: {len(all_places)}")

        all_ranked_places = rank_places(all_places, user_id)

        # Get only places that haven't been shown before
        new_places = [p for p in all_ranked_places if p["_id"] not in previously_shown_place_ids]
        print(f"New places available: {len(new_places)}")

        # If we have less than 10 new places, supplement with discovery places
        if len(new_places) < 10:
            print(f"Not enough new places, adding discovery places")

            # Get number of additional places needed
            additional_needed = 10 - len(new_places)

            # Try to get trending places first (outside user's normal patterns)
            trending_places = get_trending_places(limit=additional_needed*2)
            trending_places = [p for p in trending_places if p["_id"] not in previously_shown_place_ids
                              and p["_id"] not in [np["_id"] for np in new_places]]

            # If still not enough, get discovery places (outside user's preferences)
            if len(trending_places) < additional_needed:
                discovery_places = get_discovery_places(user_id, limit=additional_needed*2)
                discovery_places = [p for p in discovery_places if p["_id"] not in previously_shown_place_ids
                                  and p["_id"] not in [np["_id"] for np in new_places]
                                  and p["_id"] not in [tp["_id"] for tp in trending_places]]

                # Add discovery places to trending places
                trending_places.extend(discovery_places)

            # If still not enough, try to refresh some old places
            if len(trending_places) < additional_needed and previously_shown_place_ids:
                print("Still not enough places, refreshing some previously shown places")
                refreshed_places = refresh_shown_places(user_id, previously_shown_place_ids, limit=additional_needed*2)

                # Only include refreshed places not already in new_places or trending_places
                refreshed_places = [p for p in refreshed_places
                                  if p["_id"] not in [np["_id"] for np in new_places]
                                  and p["_id"] not in [tp["_id"] for tp in trending_places]]

                # Add some of the refreshed places to trending_places
                trending_places.extend(refreshed_places)

            # Take only what we need from trending places
            trending_to_use = trending_places[:additional_needed]

            # Add these to new places
            new_places.extend(trending_to_use)

            print(f"After supplementing: {len(new_places)} places available")

        # Take exactly 10 new places (or all if less than 10 are available)
        new_places_to_show = new_places[:10]
        print(f"New places to show: {len(new_places_to_show)}")

        # Get places from the last request specifically (not all history)
        last_shown_place_ids = get_last_shown_places(user_id)
        old_places = [p for p in all_ranked_places if p["_id"] in last_shown_place_ids]
        print(f"Places from last request to include: {len(old_places)}")

        # Limit old places to a maximum of 20
        old_places_to_show = old_places[:20]
        print(f"Places from last request to show (limited): {len(old_places_to_show)}")

        # Combine new places (at top) with old places (at bottom)
        final_places = new_places_to_show + old_places_to_show
        print(f"Total places in response: {len(final_places)}")

        # Update the list of shown places with just the new ones we're showing
        new_place_ids = [p["_id"] for p in new_places_to_show]
        update_shown_places(user_id, new_place_ids, max_places=None)  # No maximum - track all shown places

        return {"user_id": user_id, "recommendations": final_places}
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return {"error": f"Error generating recommendations: {str(e)}"}

# ✅ API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Travel Recommendation API",
        "version": "2.0",
        "status": "active",
        "features": [
            "Content-based recommendations",
            "Collaborative filtering",
            "Trending places",
            "Discovery mode",
            "Enhanced caching system"
        ],
        "endpoints": {
            "recommendations": "/recommendations/{user_id}",
            "cache_status": "/cache/status/{user_id}",
            "clear_cache": "/cache/{user_id}",
            "force_generation": "/cache/generate/{user_id}",
            "cache_stats": "/cache/stats"
        }
    }

@app.get("/db-status")
async def db_status():
    """Check MongoDB connection status"""
    try:
        # Ping the database
        client.admin.command('ping')
        return {"status": "connected", "message": "Successfully connected to MongoDB"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    try:
        # Count total cache entries
        total_entries = db.recommendation_cache.count_documents({})
        
        # Count active locks
        active_locks = db.cache_locks.count_documents({})
        
        # Get counts by user
        pipeline = [
            {"$group": {"_id": "$user_id", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}  # Top 10 users by cache count
        ]
        
        user_stats = list(db.recommendation_cache.aggregate(pipeline))
        
        # Get oldest and newest cache entries
        oldest = list(db.recommendation_cache.find().sort("timestamp", 1).limit(1))
        newest = list(db.recommendation_cache.find().sort("timestamp", -1).limit(1))
        
        oldest_time = oldest[0]["timestamp"] if oldest else None
        newest_time = newest[0]["timestamp"] if newest else None
        
        # Get current time for reference
        now = datetime.now()
        
        # Calculate age of oldest entry
        age_minutes = None
        if oldest_time:
            age_seconds = (now - oldest_time).total_seconds()
            age_minutes = round(age_seconds / 60, 2)
        
        return {
            "total_cache_entries": total_entries,
            "active_generation_locks": active_locks,
            "top_users": [{"user_id": stats["_id"], "cache_count": stats["count"]} for stats in user_stats],
            "oldest_entry_time": oldest_time,
            "newest_entry_time": newest_time,
            "oldest_entry_age_minutes": age_minutes,
            "shown_places_count": db.user_shown_places.count_documents({})
        }
    except Exception as e:
        return {"error": f"Error getting cache stats: {str(e)}"}

@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: str, background_tasks: BackgroundTasks):
    """Get recommendations for a user with caching support"""
    try:
        # Get all cached recommendations for this user, sorted by sequence
        cached_entries = get_user_cached_recommendations(user_id)
        
        print(f"User {user_id} has {len(cached_entries)} cached recommendation sets")
        
        # If we have cached entries, use the first one
        if cached_entries:
            entry_to_use = cached_entries[0]
            sequence = entry_to_use.get("sequence", 0)
            print(f"Cache hit for user {user_id}, using sequence {sequence}")
            
            # Remove this entry from the cache
            db.recommendation_cache.delete_one({"_id": entry_to_use["_id"]})
            
            # If we're running low on cached entries, schedule replenishment
            remaining_count = len(cached_entries) - 1
            if remaining_count <= 3:
                print(f"Running low on cache for user {user_id}, scheduling background replenishment")
                
                # Only schedule if not already generating
                if not db.cache_locks.find_one({"key": f"generating_{user_id}"}):
                    background_tasks.add_task(
                        background_cache_recommendations, 
                        user_id=user_id,
                        count=7 - remaining_count  # Top up to 7 total
                    )
            
            # Return the cached recommendations
            return entry_to_use["recommendations"]
        
        # No cache hit, generate recommendations now
        print(f"No cache for user {user_id}, generating recommendations")
        recommendations = generate_final_recommendations(user_id)
        
        # Store the first result in cache with sequence 0
        store_cache_entry(user_id, recommendations, 0)
        
        # Generate additional recommendations in the background if no lock exists
        if not db.cache_locks.find_one({"key": f"generating_{user_id}"}):
            print(f"Scheduling background generation of additional recommendations")
            background_tasks.add_task(
                background_cache_recommendations, 
                user_id=user_id,
                count=6  # Generate 6 more sets
            )
        
        return recommendations
    except Exception as e:
        print(f"Error processing recommendation request: {e}")
        return {"error": f"Error processing request: {str(e)}"}

@app.get("/cache/status/{user_id}")
def get_cache_status(user_id: str):
    """Check cache status for a user"""
    try:
        # Get all cached entries for this user
        cached_entries = get_user_cached_recommendations(user_id)
        
        # Check if background generation is in progress
        is_generating = False
        lock = db.cache_locks.find_one({"key": f"generating_{user_id}"})
        if lock:
            is_generating = True
            lock_time = lock.get("timestamp")
        
        # Expiry information
        expiry_info = {}
        if cached_entries:
            oldest_entry = cached_entries[0]  # Already sorted by sequence
            created_time = oldest_entry.get("timestamp")
            if created_time:
                # Calculate expiry time (6 hours from creation)
                expiry_time = created_time + timedelta(hours=6)
                now = datetime.now()
                remaining_time = (expiry_time - now).total_seconds() / 60  # minutes
                
                expiry_info = {
                    "created": created_time,
                    "expires": expiry_time,
                    "remaining_minutes": max(0, round(remaining_time, 2))
                }
        
        # Get sequences as list
        sequences = [entry.get("sequence") for entry in cached_entries]
        
        return {
            "user_id": user_id,
            "cache_entries": len(cached_entries),
            "sequences": sequences,
            "has_cache": len(cached_entries) > 0,
            "is_generating": is_generating,
            "generation_started_at": lock_time if is_generating else None,
            "expiry": expiry_info
        }
    except Exception as e:
        return {"error": f"Error checking cache status: {str(e)}"}

@app.delete("/cache/{user_id}")
def clear_user_cache_endpoint(user_id: str, reset_shown: bool = False):
    """Clear cache and optionally shown places for a user"""
    try:
        # Clear the cache
        deleted = clear_user_cache(user_id)
        
        # Also clear any locks for this user
        lock_deleted = db.cache_locks.delete_one({"key": f"generating_{user_id}"}).deleted_count
        
        # Optionally reset shown places
        shown_reset = False
        if reset_shown:
            shown_reset = reset_user_shown_places(user_id)
        
        return {
            "message": f"Cleared {deleted} cache entries for user {user_id}",
            "deleted_count": deleted,
            "lock_cleared": lock_deleted > 0,
            "shown_places_reset": shown_reset
        }
    except Exception as e:
        return {"error": f"Error clearing cache: {str(e)}"}

@app.post("/cache/generate/{user_id}")
async def force_cache_generation(user_id: str, background_tasks: BackgroundTasks, count: int = 6):
    """Force immediate cache generation for a user"""
    try:
        # Check if already generating
        if db.cache_locks.find_one({"key": f"generating_{user_id}"}):
            return {
                "message": f"Cache generation already in progress for user {user_id}",
                "status": "in_progress"
            }
        
        # Schedule immediate generation
        background_tasks.add_task(
            background_cache_recommendations,
            user_id=user_id,
            count=count
        )
        
        return {
            "message": f"Scheduled cache generation for user {user_id}",
            "entries_to_generate": count,
            "status": "scheduled"
        }
    except Exception as e:
        return {"error": f"Error scheduling cache generation: {str(e)}"}

@app.delete("/shown-places/{user_id}")
def reset_shown_places_endpoint(user_id: str):
    """Reset shown places tracking for a user"""
    try:
        success = reset_user_shown_places(user_id)
        return {
            "message": f"Reset shown places tracking for user {user_id}",
            "success": success
        }
    except Exception as e:
        return {"error": f"Error resetting shown places: {str(e)}"}

@app.get("/search/{user_id}")
async def search_user(
    user_id: str,
    query: str = Query(None, min_length=1),
    limit: int = Query(10, ge=1, le=50)
):
    """Search for places with keyword matching"""
    if not query:
        return {"user_id": user_id, "query": "", "results": []}
        
    query_lower = query.lower()
    
    # Get all places
    places = list(db.places.find({}))
    
    # Score places based on keyword matching
    scored_places = []
    for place in places:
        name = place.get("name", "").lower()
        desc = place.get("description", "").lower()
        
        # Calculate basic text matching score
        name_score = 1.0 if query_lower in name else 0.0
        desc_score = 0.5 if query_lower in desc else 0.0
        
        # Combined score
        score = name_score + desc_score
        
        if score > 0:
            scored_places.append({
                "place_id": place.get("_id", ""),
                "name": place.get("name", ""),
                "score": score,
                "data": place
            })
    
    # Sort by score and return top results
    scored_places.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "user_id": user_id,
        "query": query,
        "results": scored_places[:limit]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
