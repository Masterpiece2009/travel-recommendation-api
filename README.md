<h1 align="center" style="font-size:3em;font-weight:900;"> <b>Explore - Travel Recommendation API</b> </h1> <div align="center"><div align="center"> <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI"/> <img src="https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white" alt="MongoDB"/> <img src="https://img.shields.io/badge/Railway-0B0D0E?style=for-the-badge&logo=railway&logoColor=white" alt="Railway"/> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/> </div> <hr style="height:1px;border:none;color:#333;background-color:#333;" />
📌 Overview
This API provides sophisticated travel recommendations and roadmap generation using NLP and hybrid filtering algorithms.

<hr style="height:1px;border-width:0;color:gray;background-color:gray">
🧠 Core Algorithms
1️⃣ Recommendation Algorithm
<p><em>The recommendation system uses a hybrid approach that combines multiple scoring methods:</em></p>
Component	Weight	Description
Category Matching	60%	Matches places based on user's preferred categories and tags
Semantic Search	40%	Uses NLP to find places that match user interests semantically
Rating Score	40%	Considers the rating of places from user reviews
Likes Score	30%	Factors in popularity based on likes
User Interactions	30%	Personalizes recommendations based on past interactions
<div align="center"> <h4>⚡ Caching Strategy ⚡</h4> </div>
⏱️ 6-hour TTL for cached recommendations
🔄 Stores 7 recommendation responses per user
🔍 Ensures new recommendations include both new places and places from recent requests
🚀 Uses background tasks to generate cache entries asynchronously
<div align="center"> <code>Key function: generate_final_recommendations()</code> </div> <hr style="height:1px;border-width:0;color:gray;background-color:gray">
2️⃣ Roadmap Generation Algorithm
<p><em>The roadmap generation uses a hybrid filtering approach with two phases:</em></p>
Phase 1: Critical Filtering (Hard Constraints)
📍 Location-based filtering to match user's preferred destinations
✅ Ensures basic compatibility with user's core requirements
Phase 2: Soft Constraint Scoring
Constraint	Weight	Description
Budget compatibility	30%	Matches places to user's budget level
Accessibility needs	20%	Ensures places meet accessibility requirements
Group type suitability	30%	Optimizes for family, solo, couples, etc.
Seasonal relevance	20%	Considers time of year for recommendations
<div align="center"> <h4>🧩 Intelligent Caching 🧩</h4> </div>
🔄 Only regenerates roadmaps when user preferences change
🔑 Uses a hash of user preferences to determine if regeneration is needed
🗺️ Includes geographical distance calculations for optimizing travel routes
<div align="center"> <code>Key function: generate_hybrid_roadmap()</code> </div> <hr style="height:1px;border-width:0;color:gray;background-color:gray">
🚀 API Features
<div style="display: flex; flex-wrap: wrap; justify-content: space-around;"> <div style="flex: 0 0 45%; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 8px;"> <h3 align="center">🔐 Security</h3> <p>Secure MongoDB connection with environment variable configuration</p> </div> <div style="flex: 0 0 45%; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 8px;"> <h3 align="center">⚡ Performance</h3> <p>FastAPI endpoints with background task processing</p> </div> <div style="flex: 0 0 45%; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 8px;"> <h3 align="center">🔄 Reliability</h3> <p>Fallback mechanisms for NLP functionality</p> </div> <div style="flex: 0 0 45%; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 8px;"> <h3 align="center">⏱️ Efficiency</h3> <p>Caching strategies with TTL indices</p> </div> </div> <hr style="height:1px;border-width:0;color:gray;background-color:gray">
📦 Deployment
The API is deployed on Railway with proper database connection and error handling.

📂 Project Structure
│
├── 📄 main.py            # Main application file with full functionality
├── 📄 main_fixed.py      # Simplified version for deployment
├── 📄 requirements.txt   # Dependencies including geopy
├── 📄 runtime.txt        # Specifies Python 3.10
└── 📄 README.md          # Documentation
<hr style="height:1px;border-width:0;color:gray;background-color:gray"> <div align="center"> <h3>🌟 Developed for improved travel recommendations and roadmap generation 🌟</h3> </div>


abdelrahmanaly
 - Travel Recommendation API
<hr>
📌 Overview
This API provides sophisticated travel recommendations and roadmap generation using NLP and hybrid filtering algorithms.

<hr>
🧠 Core Algorithms
1️⃣ Recommendation Algorithm
<p><em>The recommendation system uses a hybrid approach that combines multiple scoring methods:</em></p>
Component	Weight	Description
Category Matching	60%	Matches places based on user's preferred categories and tags
Semantic Search	40%	Uses NLP to find places that match user interests semantically
Rating Score	40%	Considers the rating of places from user reviews
Likes Score	30%	Factors in popularity based on likes
User Interactions	30%	Personalizes recommendations based on past interactions
<div align="center"> <h4>⚡ Caching Strategy ⚡</h4> </div>
⏱️ 6-hour TTL for cached recommendations
🔄 Stores 7 recommendation responses per user
🔍 Ensures recommendations include new places and places from recent requests
🚀 Uses background tasks to generate cache entries asynchronously
<div align="center"> <code>Key function: generate_final_recommendations()</code> </div> <hr>
2️⃣ Roadmap Generation Algorithm
<p><em>The roadmap generation uses a hybrid filtering approach with two phases:</em></p>
Phase 1: Critical Filtering (Hard Constraints)
📍 Location-based filtering to match user's preferred destinations
✅ Ensures basic compatibility with user's core requirements
Phase 2: Soft Constraint Scoring
Constraint	Weight	Description
Budget compatibility	30%	Matches places to user's budget level
Accessibility needs	20%	Ensures places meet accessibility requirements
Group type suitability	30%	Optimizes for family, solo, couples, etc.
Seasonal relevance	20%	Considers time of year for recommendations
<div align="center"> <h4>🧩 Intelligent Caching 🧩</h4> </div>
🔄 Only regenerates roadmaps when user preferences change
🔑 Uses a hash of user preferences to determine if regeneration is needed
🗺️ Includes geographical distance calculations for route optimization
<div align="center"> <code>Key function: generate_hybrid_roadmap()</code> </div> <hr>
🚀 API Features
<table width="100%"> <tr> <td align="center" width="25%"> <h3>🔐 Security</h3> <p>Secure MongoDB connection with environment variables</p> </td> <td align="center" width="25%"> <h3>⚡ Performance</h3> <p>FastAPI endpoints with background task processing</p> </td> <td align="center" width="25%"> <h3>🔄 Reliability</h3> <p>Fallback mechanisms for NLP functionality</p> </td> <td align="center" width="25%"> <h3>⏱️ Efficiency</h3> <p>Caching strategies with TTL indices</p> </td> </tr> </table> <hr>
📦 Deployment
The API is deployed on Railway with proper database connection and error handling.

📂 Project Structure
│
├── 📄 main.py            # Main application with full functionality
│
├── 📄 main_fixed.py      # Simplified version for deployment
│
├── 📄 requirements.txt   # Dependencies including geopy
│
├── 📄 runtime.txt        # Specifies Python 3.10
│
└── 📄 README.md          # Documentation
<hr> <div align="center"> <h3>🌟 Developed for improved travel recommendations and roadmap generation 🌟</h3>  </div>
Key Functions
Core Recommendation System
load_spacy_model()
Loads the spaCy model with word vectors for semantic similarity calculations.

def load_spacy_model():
    """
    Loads the spaCy model with word vectors for semantic similarity.
    Returns the model or a DummyNLP fallback if loading fails.
    
    Returns:
        spaCy model or DummyNLP fallback
    """
Example:

nlp = load_spacy_model()
if not isinstance(nlp, DummyNLP):
    # Process using spaCy word vectors
    similarity = nlp("beach").similarity(nlp("ocean"))
get_candidate_places(search_query, top_n=20)
Retrieves candidate places based on semantic similarity with the search query.

def get_candidate_places(search_query, top_n=20):
    """
    Retrieves candidate places based on semantic similarity with search query.
    Uses NLP to match search terms with place tags and categories.
    
    Args:
        search_query: User search query
        top_n: Number of top candidates to return
        
    Returns:
        List of candidate places sorted by relevance
    """
Example:

# Finding places semantically related to "historic landmarks"
candidates = get_candidate_places("historic landmarks")
# Returns: [{'_id': 'place003', 'name': 'Mont Saint-Michel', ...}, ...]
generate_final_recommendations(user_id, search_query=None, n=10)
Generates final personalized recommendations by combining semantic search, user history, and collaborative filtering.

def generate_final_recommendations(user_id, search_query=None, n=10):
    """
    Generates personalized recommendations by combining semantic search,
    user history, and collaborative filtering.
    
    Args:
        user_id: User ID
        search_query: Optional search query
        n: Number of recommendations to return
        
    Returns:
        List of personalized place recommendations
    """
Example:

# Get personalized recommendations for user005
recommendations = generate_final_recommendations("user005", "historic sites")
# Returns: [
#   {
#     "place": {"_id": "place003", "name": "Mont Saint-Michel", ...},
#     "score": 0.85,
#     "match_reason": "Matches your interest in historical architecture"
#   },
#   ...
# ]
Roadmap Generation
generate_hybrid_roadmap(user_id)
Generates a personalized travel roadmap using a two-stage filtering approach based on user preferences.

def generate_hybrid_roadmap(user_id):
    """
    Generate a travel roadmap for a user using a hybrid two-stage filtering approach.
    First applies critical filters, then soft constraints with weighted scoring.
    
    Args:
        user_id: User ID
        
    Returns:
        Dictionary containing roadmap data
    """
Example:

# Generate travel roadmap for user005
roadmap = generate_hybrid_roadmap("user005")
# Returns: {
#   "start_date": "April 2024",
#   "budget_level": "low",
#   "group_type": "couple",
#   "places": [
#     {"name": "Mont Saint-Michel", "match_scores": {"total": 0.85, ...}},
#     ...
#   ],
#   "routes": [
#     {"from": "place003", "to": "place007", "type": "direct"},
#     ...
#   ]
# }
get_roadmap_with_caching(user_id)
Retrieves or generates a roadmap with intelligent caching based on user preferences.

def get_roadmap_with_caching(user_id):
    """
    Retrieves or generates a roadmap with intelligent caching.
    Only regenerates the roadmap when user preferences have changed.
    
    Args:
        user_id: User ID
        
    Returns:
        Dictionary containing roadmap data
    """
Example:

# Get cached roadmap or generate new one if needed
roadmap = get_roadmap_with_caching("user005")
# Returns cached roadmap if preferences haven't changed
# or generates new one if they have
Cache and Data Management
get_recommendations_with_caching(user_id, search_query=None, n=10)
Retrieves or generates recommendations with TTL-based caching.

def get_recommendations_with_caching(user_id, search_query=None, n=10):
    """
    Retrieves or generates recommendations with TTL-based caching.
    Caches 7 most recent responses per user to avoid recomputing.
    
    Args:
        user_id: User ID
        search_query: Optional search query
        n: Number of recommendations to return
        
    Returns:
        List of place recommendations
    """
Example:

# Get cached recommendations or generate new ones
recommendations = get_recommendations_with_caching("user005", "beaches")
# Returns cached results if available and valid
# or generates new ones if cache miss or expired
update_user_interaction(user_id, place_id, interaction_type='view')
Updates user interaction history for a place.

def update_user_interaction(user_id, place_id, interaction_type='view'):
    """
    Updates user interaction history for a place.
    Tracks views, likes, and bookmarks for recommendation tuning.
    
    Args:
        user_id: User ID
        place_id: Place ID
        interaction_type: Type of interaction ('view', 'like', 'bookmark')
        
    Returns:
        Updated interaction document
    """
Example:

# Record that user005 liked place003
update_user_interaction("user005", "place003", "like")
# Returns: {"user_id": "user005", "place_id": "place003", "type": "like", ...}
Helper Functions
parse_travel_dates(travel_dates_str)
Extracts the month from a travel date string.

def parse_travel_dates(travel_dates_str):
    """
    Extract month from travel dates string like "April 2024".
    
    Args:
        travel_dates_str: String containing travel dates
        
    Returns:
        String containing the month name
    """
Example:

# Extract month from travel dates
month = parse_travel_dates("April 2024")
# Returns: "April"
check_accessibility_compatibility(place, accessibility_needs)
Checks if a place meets the user's accessibility needs.

def check_accessibility_compatibility(place, accessibility_needs):
    """
    Checks if a place meets the user's accessibility needs.
    
    Args:
        place: Place document
        accessibility_needs: List of required accessibility features
        
    Returns:
        Boolean indicating compatibility
    """
Example:

# Check if place meets senior-friendly accessibility needs
is_compatible = check_accessibility_compatibility(
    {"name": "Mont Saint-Michel", "accessibility": ["wheelchair-accessible"]},
    ["senior-friendly"]
)
# Returns: False (doesn't meet the specific need)
API Endpoints
/api/recommendations/{user_id}
Get personalized recommendations for a user.

Query Parameters:

search: Optional search query
n: Number of recommendations (default: 10)
Response:

{
  "recommendations": [
    {
      "place": {
        "_id": "place003",
        "name": "Mont Saint-Michel",
        "category": "historical",
        "tags": ["UNESCO", "architecture", "scenic"]
      },
      "score": 0.85,
      "match_reason": "Matches your interest in historical architecture"
    }
  ],
  "generated_at": "2025-04-12T14:30:00Z",
  "cache_status": "HIT"
}
/api/roadmap/{user_id}
Get a personalized travel roadmap for a user.

Response:

{
  "start_date": "April 2024",
  "budget_level": "low",
  "group_type": "couple",
  "places": [
    {
      "_id": "place003",
      "name": "Mont Saint-Michel",
      "match_scores": {
        "total": 0.85,
        "budget": 0.7,
        "accessibility": 0.5,
        "group": 1.0,
        "seasonal": 1.0
      }
    }
  ],
  "routes": [
    {
      "from": "place003",
      "to": "place007",
      "from_name": "Mont Saint-Michel",
      "to_name": "Louvre Museum",
      "type": "direct"
    }
  ],
  "cache_status": "GENERATED"
}
/api/interaction/{user_id}/{place_id}
Record a user interaction with a place.

Query Parameters:

type: Interaction type ("view", "like", "bookmark")
Response:

{
  "success": true,
  "user_id": "user005",
  "place_id": "place003",
  "interaction_type": "like",
  "timestamp": "2025-04-12T14:35:00Z"
}
Data Structures
Place Document
{
  "_id": "place003",
  "name": "Mont Saint-Michel",
  "category": "historical",
  "tags": ["UNESCO", "architecture", "scenic"],
  "description": "A breathtaking medieval island monastery, one of France's most iconic landmarks.",
  "location": {
    "city": "Normandy",
    "country": "France",
    "latitude": 48.6361,
    "longitude": -1.5115
  },
  "accessibility": [],
  "average_rating": 4.7,
  "likes": 7500,
  "reviews_count": 3100,
  "appropriate_time": ["April", "May", "June", "September", "October"],
  "budget": "high",
  "group_type": "family"
}
User Preferences Document
{
  "_id": "pref005",
  "user_id": "user005",
  "destinations": ["Marseille", "Provence", "Carcassonne"],
  "travel_dates": "April 2024",
  "group_type": "couple",
  "accessibility_needs": ["senior-friendly"],
  "budget": "low"
}
Algorithm Weights
Roadmap Generation
Budget compatibility: 30%
Accessibility compatibility: 20%
Group type compatibility: 30%
Seasonal compatibility: 20%
Recommendation Algorithm
Category/tag matching: 60%
Semantic search: 40%
Rating: 40%
Likes/interactions: 30%
Caching Strategy
Recommendations are cached for 6 hours using a TTL index
Each user has up to 7 cached recommendation responses
Roadmaps are cached until user preferences change
Cache entries include a timestamp for TTL calculation
