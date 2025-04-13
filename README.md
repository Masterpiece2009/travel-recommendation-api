🌍 EXPLORE - TRAVEL RECOMMENDATION API
======================================

A sophisticated travel recommendation system that delivers personalized travel suggestions and dynamic itineraries. It combines collaborative filtering, content-based filtering, semantic search, and discovery mechanisms to ensure high-quality and diverse travel experiences. Built with robust fallback strategies, intelligent caching, and efficient pagination to maximize user satisfaction.

--------------------------------------
⚙️ SYSTEM OVERVIEW
--------------------------------------

**Explore** provides advanced, personalized travel recommendations using:

- **Collaborative Filtering** (40%): Based on interactions of similar users.
- **Content-Based Filtering** (60%): Based on a user's own preferences.
- **Category Matching** (60%): Matches locations with tags and categories.
- **Semantic Search** (40%): Uses NLP to recommend semantically relevant places.
- **Discovery Mechanisms**: Prevents recommendation bubbles by suggesting diverse and fresh places.

--------------------------------------
🚀 CORE FEATURES
--------------------------------------

**Multi-Strategy Recommendation Engine**
- Combines multiple filters and weights for high-accuracy suggestions.

**Advanced Caching System**
- Two-Level NLP Caching: Reduces response time from 12s.
- TTL-based Expiry: Keeps data fresh (6-hour TTL).
- Stores 7 cached recommendations per user.
- Background generation with lock mechanisms.

**Progressive Pagination**
- 1st request: 10 new places.
- 2nd request: 10 new + 10 previous (20 total).
- 3rd+ request: 10 new + 20 previous (30 total).

--------------------------------------
🗺️ PERSONALIZED ROADMAP GENERATION
--------------------------------------

**Critical Filters (Hard Constraints)**
- Location-Based Filtering.
- Accessibility Requirements.

**Soft Constraint Scoring**
- Budget Compatibility: 30%
- Accessibility Features: 20%
- Group Type Suitability (Family, Solo, Couples): 30%
- Seasonal Relevance: 20%

**Intelligent Caching**
- Roadmaps regenerate only when preferences change.
- Uses preference hash to detect changes.

--------------------------------------
🧩 KEY FUNCTIONS
--------------------------------------

**Recommendation Generation**
- `get_candidate_places()`: Fetch candidate places matching preferences.
- `generate_final_recommendations()`: Combines all filters and fallbacks.
- `get_collaborative_recommendations()`: Uses similar users' history.

**Roadmap Generation**
- `generate_hybrid_roadmap()`: Generates itinerary with fallback support.
- `get_roadmap_with_caching()`: Fetches or regenerates roadmaps intelligently.

**Helper Functions**
- `load_spacy_model()`: Loads word vectors for NLP.
- `compute_text_similarity()`: Compares semantic similarity.
- `check_accessibility_compatibility()`: Validates accessibility.
- `calculate_budget_compatibility()`: Budget relevance score.

--------------------------------------
🔁 SYSTEM FLOW
--------------------------------------

**Recommendation Flow**
User Request  
→ Check Cache  
 ├── Collaborative Recommendations (40%)  
 └── Content-Based Recommendations (60%)

**Roadmap Generation Flow**
User Request  
→ Generate Hybrid Roadmap  
 ├── Critical Filters → Accessibility + Destination  
 ├── Weighted Scoring → Budget + Group + Season  
 └── Fallbacks → Nearby → Trending → Top-Rated

--------------------------------------
✅ RECENT OPTIMIZATIONS
--------------------------------------

- Modified collaborative recommendations to return place IDs.
- Implemented two-level NLP caching to reduce processing load.
- Enhanced similar users caching (12-hour expiration).
- Improved timezone comparison for datetimes.
- Implemented lazy-loading to minimize DB queries.

--------------------------------------
🗃️ DATABASE COLLECTIONS & TTL
--------------------------------------

**Core Collections (Permanent):**
- `users`
- `places`
- `interactions`

**Cache Collections (With TTL):**
- `recommendations_cache`: 6h TTL
- `roadmaps`: 24h TTL
- `cache_locks`: 10m TTL
- `shown_places`: 6h TTL
- `user_keywords_cache`: 24h TTL
- `similar_users_cache`: 12h TTL

--------------------------------------
📁 PROJECT STRUCTURE
--------------------------------------

main_fixed.py        → Deployment version  
requirements.txt     → Dependencies (spaCy, geopy, etc.)  
runtime.txt          → Python version (3.10)  
README.md            → Documentation

--------------------------------------
🚀 FUTURE ENHANCEMENTS
--------------------------------------

- Tensor Factorization for better collaborative filtering.
- Real-time Interaction Processing via WebSockets.
- Natural Language Query Processing for recommendations.
- User Clustering for improved user similarity.

--------------------------------------
🌟 BUILT TO MAKE DISCOVERING NEW DESTINATIONS SMARTER, FASTER, AND MORE PERSONAL 🌟
