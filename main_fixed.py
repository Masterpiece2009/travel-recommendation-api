import os
import datetime
import asyncio
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
import pymongo
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
    """Update the list of places shown to a user"""
    try:
        current_time = datetime.datetime.utcnow()
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
    """Get the list of places most recently shown to a user"""
    try:
        user_data = db.user_shown_places.find_one({"user_id": user_id})
        if user_data and "place_ids" in user_data:
            return user_data["place_ids"]
        return []
    except Exception as e:
        print(f"Error getting shown places for user {user_id}: {e}")
        return []

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
                "place_id": place.get("place_id", ""),
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
    uvicorn.run("main_fixed:app", host="0.0.0.0", port=port)
