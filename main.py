import os
import json
import time
import asyncio
import datetime
import random
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
import pymongo
from sklearn.metrics.pairwise import cosine_similarity
import nest_asyncio
import uvicorn


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
    db = client.travel_recommendations
else:
    raise ValueError("MongoDB URI or password not provided in environment variables")
# TTL in seconds (6 hours)
TTL_SECONDS = 6 * 60 * 60

def initialize_collections():
    """Initialize all MongoDB collections with proper indexes"""
    try:
        # Initialize recommendation_cache collection with TTL index
        cache_collection = db.recommendation_cache
        cache_collection.create_index(
            "timestamp", 
            expireAfterSeconds=TTL_SECONDS
        )
        
        # Initialize cache_locks collection
        locks_collection = db.cache_locks
        
        # Initialize user_shown_places collection with TTL index
        shown_places_collection = db.user_shown_places
        shown_places_collection.create_index(
            "timestamp", 
            expireAfterSeconds=TTL_SECONDS
        )
        
        print("Collections initialized successfully with TTL indexes")
        return True
    except Exception as e:
        print(f"Error initializing collections: {e}")
        return False

# Initialize collections on startup
initialize_collections()
async def update_shown_places(user_id: str, place_ids: List[str]):
    """
    Update the list of places shown to a user
    Each update replaces the previous list and resets the TTL
    """
    try:
        # Current timestamp for TTL
        current_time = datetime.datetime.utcnow()
        
        # Update or insert user shown places
        db.user_shown_places.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "place_ids": place_ids,
                    "timestamp": current_time
                }
            },
            upsert=True
        )
        return True
    except Exception as e:
        print(f"Error updating shown places for user {user_id}: {e}")
        return False

async def get_last_shown_places(user_id: str) -> List[str]:
    """
    Get the list of places most recently shown to a user
    """
    try:
        user_data = db.user_shown_places.find_one({"user_id": user_id})
        if user_data and "place_ids" in user_data:
            return user_data["place_ids"]
        return []
    except Exception as e:
        print(f"Error getting shown places for user {user_id}: {e}")
        return []

async def reset_user_shown_places(user_id: str):
    """
    Reset (clear) the list of places shown to a user
    """
    try:
        db.user_shown_places.delete_one({"user_id": user_id})
        return True
    except Exception as e:
        print(f"Error resetting shown places for user {user_id}: {e}")
        return False
async def generate_final_recommendations(user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
    """
    Generate final recommendations combining new places and previously shown places
    Returns up to 10 new places followed by up to 20 places from most recent request
    """
    # Places collection (sample data structure)
    places = list(db.places.find({}))
    
    # Get user preferences
    user = db.users.find_one({"user_id": user_id}) or {"preferences": {}}
    
    # Get last shown places
    last_shown_places = await get_last_shown_places(user_id)
    
    # Compute recommendations based on various factors
    # 1. Category matching (60% weight)
    # 2. Semantic search (40% weight)
    # 3. Rating (40% weight)
    # 4. Likes (30% weight)
    # 5. Interactions (30% weight)
    
    # Calculate scores (simplified for example)
    scored_places = []
    for place in places:
        # Skip places already shown to the user
        if place["place_id"] in last_shown_places:
            continue
            
        # Sample scoring algorithm
        score = (
            0.6 * random.random() +  # Category matching placeholder
            0.4 * random.random() +  # Semantic search placeholder
            0.4 * (place.get("rating", 0) / 5) +  # Rating (normalized to 0-1)
            0.3 * min(place.get("likes", 0) / 1000, 1) +  # Likes (capped at 1000)
            0.3 * random.random()  # User interactions placeholder
        )
        
        scored_places.append({
            "place_id": place["place_id"],
            "name": place["name"],
            "score": score,
            "data": place
        })
    
    # Sort by score and get top 10 new places
    scored_places.sort(key=lambda x: x["score"], reverse=True)
    new_places = scored_places[:10]
    
    # Get previously shown places (from last request only)
    previous_places = []
    if last_shown_places:
        for place_id in last_shown_places[:20]:  # Limited to top 20
            place = db.places.find_one({"place_id": place_id})
            if place:
                previous_places.append({
                    "place_id": place["place_id"],
                    "name": place["name"],
                    "score": 0.5,  # Lower priority
                    "data": place
                })
    
    # Combine new places with previous places
    final_recommendations = new_places + previous_places
    
    # Update shown places with new recommendations
    all_place_ids = [rec["place_id"] for rec in final_recommendations]
    await update_shown_places(user_id, all_place_ids)
    
    # Return recommendations up to the limit
    return final_recommendations[:limit]

async def background_cache_recommendations(user_id: str):
    """Background task to cache recommendations"""
    lock_key = f"cache_lock_{user_id}"
    
    # Try to acquire lock
    lock_result = db.cache_locks.update_one(
        {"_id": lock_key, "locked": {"$ne": True}},
        {"$set": {"locked": True, "timestamp": datetime.datetime.utcnow()}}
    )
    
    # Return if lock acquisition failed
    if lock_result.modified_count == 0:
        return
    
    try:
        # Generate recommendations
        recommendations = await generate_final_recommendations(user_id)
        
        # Store in cache
        db.recommendation_cache.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "recommendations": recommendations,
                    "timestamp": datetime.datetime.utcnow()
                }
            },
            upsert=True
        )
    finally:
        # Release lock
        db.cache_locks.update_one(
            {"_id": lock_key},
            {"$set": {"locked": False}}
        )
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Travel Recommendation API",
        "version": "1.0",
        "status": "active"
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

@app.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: str, 
    background_tasks: BackgroundTasks,
    limit: int = Query(30, ge=1, le=100)
):
    """Get recommendations for a user"""
    # Check cache first
    cached = db.recommendation_cache.find_one({"user_id": user_id})
    
    # If cache exists, return it
    if cached and "recommendations" in cached:
        # Check if we need to replenish the cache (if only 3 or fewer entries left)
        cache_count = db.recommendation_cache.count_documents({"user_id": user_id})
        if cache_count <= 3:
            background_tasks.add_task(background_cache_recommendations, user_id)
        
        return {
            "user_id": user_id,
            "recommendations": cached["recommendations"][:limit],
            "source": "cache",
            "timestamp": cached["timestamp"]
        }
    
    # If no cache, generate recommendations
    recommendations = await generate_final_recommendations(user_id, limit)
    
    # Schedule background task to cache more recommendations
    background_tasks.add_task(background_cache_recommendations, user_id)
    
    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "source": "generated",
        "timestamp": datetime.datetime.utcnow()
    }

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    stats = {
        "cache_count": db.recommendation_cache.count_documents({}),
        "users_with_cache": len(db.recommendation_cache.distinct("user_id")),
        "locks_count": db.cache_locks.count_documents({"locked": True}),
        "shown_places_count": db.user_shown_places.count_documents({})
    }
    return stats

@app.delete("/cache/{user_id}")
async def clear_user_cache(user_id: str):
    """Clear cache for a specific user"""
    # Delete cache entries
    cache_result = db.recommendation_cache.delete_many({"user_id": user_id})
    
    # Reset shown places
    await reset_user_shown_places(user_id)
    
    return {
        "user_id": user_id,
        "cache_entries_deleted": cache_result.deleted_count,
        "shown_places_reset": True
    }

# Replace:
@app.get("/search/{user_id}")
async def search_user(
    user_id: str,
    query: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50)
):
    """Search for places with semantic search"""
    # Process the query with spaCy
    query_doc = nlp(query)
    
    # ... rest of the function ...

@app.get("/search/{user_id}")
async def search_user(
    user_id: str,
    query: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50)
):
    """Search for places with keyword matching"""
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
        
        scored_places.append({
            "place_id": place["place_id"],
            "name": place["name"],
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
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
