🌍 Explore - Travel Recommendation API
📋 System Overview
Our sophisticated travel recommendation system provides personalized travel suggestions and itineraries using advanced algorithms combining collaborative filtering, content-based filtering, and discovery mechanisms. The system includes multiple fallback strategies ensuring robust recommendations even when primary methods fail.

Unsupported image
 
Unsupported image
 
Unsupported image
 
Unsupported image

💡 Core Features
🔄 Multi-Strategy Recommendation Engine
Collaborative filtering (40% weight) - Based on similar users' interactions
Content-based filtering (60% weight) - Based on user's historical preferences
Category Matching (60%) - Matches places based on user's preferred categories/tags
Semantic Search (40%) - Uses NLP to find places that match user interests semantically
Discovery mechanisms - Prevents recommendation bubbles with diverse suggestions
🚀 Advanced Caching System
Background generation with locking mechanisms
Two-level NLP caching - Reduces processing time from 12 seconds
TTL-based cache expiration - Ensures fresh content
6-hour TTL for cached recommendations
Stores 7 recommendation responses per user
Progressive pagination system that properly tracks shown places
📱 Progressive Pagination
First request: 10 new places
Second request: 10 new + 10 previous (20 total)
Third+ request: 10 new + 20 previous (30 total)
🗺️ Personalized Roadmap Generation
Two-stage filtering approach:

Critical Filters (Hard Constraints)
Location-based filtering for preferred destinations
Accessibility needs compatibility
Soft Constraint Scoring
Budget compatibility (30%) - Matches places to user's budget level
Accessibility features (20%) - Ensures places meet accessibility requirements
Group type suitability (30%) - Optimizes for family, solo, couples, etc.
Seasonal relevance (20%) - Considers time of year for recommendations
Intelligent Caching

Only regenerates roadmaps when user preferences change
Uses a hash of user preferences to determine if regeneration is needed
Includes geographical distance calculations for route optimization
⚙️ Key Functions
📊 Recommendation Generation
get_candidate_places(user_prefs, user_id, size=100)

Retrieves candidate places matching user preferences
Combines direct category/tag matching with NLP-based similarity
Falls back to popularity-based recommendations if needed
generate_final_recommendations(user_id, num_recommendations=10, previously_shown_ids=None)

Combines collaborative (40%) and content-based (60%) filtering
Implements 4 fallback mechanisms:
Discovery places - Outside user patterns
Trending places - Popular in recent interactions
Rediscovery - Places from beginning of user history
Top-rated places - Absolute last resort
Always returns exactly the requested number of recommendations
get_collaborative_recommendations(user_id, target_count=10, excluded_place_ids=None)

Provides recommendations based on similar users' interactions
Implements similarity user caching for performance
Uses weighted interaction scoring with time decay
Returns place IDs for compatibility with other functions
🗺️ Roadmap Generation
generate_hybrid_roadmap(user_id)

Creates comprehensive travel itineraries for users
Uses two-stage filtering approach
Implements 3-tier fallback to ensure exactly 10 places:
Nearby places (within 100km)
Trending places (popular in last 14 days)
Top-rated places (last resort)
Generates sequential routes between places
get_roadmap_with_caching(user_id)

Retrieves or generates a roadmap with intelligent caching
Only regenerates when user preferences have changed
Uses a hash of preferences to determine if regeneration is needed
🛠️ Helper Functions
load_spacy_model(model="en_core_web_md", retries=2)

Loads spaCy model with word vectors for semantic similarity
Implements fallback to simpler models if full model unavailable
Provides robust text matching even when vectors unavailable
compute_text_similarity(text1, text2)

Computes similarity between text strings
Uses word vectors when available, falls back to word overlap
check_accessibility_compatibility(place, accessibility_needs)

Verifies if a place meets user's accessibility requirements
Returns true if no accessibility needs specified
calculate_budget_compatibility(place_budget_level, user_budget_level)

Computes normalized compatibility score (0-1) between budgets
Linear penalty based on budget level differences
🔄 System Flow
Recommendation Flow

User Request → Check Cache → Generate Recommendations:
  ├── Collaborative Recommendations (40%)
  │   └── Similar users → Score interactions → Top places
  └── Content-Based Recommendations (60%)
      └── Candidate places → Personalization scoring → Top places
Roadmap Generation Flow

User Request → Generate Hybrid Roadmap:
  ├── Critical Filters → Accessibility + Destination
  ├── Weighted Scoring → Budget + Group + Season
  └── Fallbacks → Nearby → Trending → Top-Rated
Caching System

Check Cache → Acquire Lock → Generate → Store → Release
                      ↑                     ↓
                      ← Background Regeneration ←
📈 Recent Optimizations
✅ Modified collaborative recommendations - Returns place IDs for compatibility
✅ Implemented two-level NLP caching - Reduced 12-second processing bottleneck
✅ Enhanced similar users caching - 12-hour expiration for performance
✅ Improved datetime handling - Fixed timezone comparison issues
✅ Implemented lazy-loading - Reduced database queries and memory usage

📦 Database Collections & Expiration
Core Data Collections (permanent storage):

users - User profiles and preferences
places - Travel destinations and attributes
interactions - User-place interaction records
Caching Collections (with TTL expiration):

recommendations_cache - Cached recommendation results (6h)
roadmaps - Generated travel itineraries (24h)
cache_locks - Locking mechanism (10m)
shown_places - Record of places shown to users (6h)
user_keywords_cache - Cached user keywords for NLP (24h)
keyword_similarity_cache - Cached keyword similarities (24h)
similar_users_cache - Cached similar users (12h)
📂 Project Structure
│
├── 📄 main.py            # Main application with full functionality
│
├── 📄 main_fixed.py      # Simplified version for deployment
│
├── 📄 requirements.txt   # Dependencies including geopy and spaCy
│
├── 📄 runtime.txt        # Specifies Python 3.10
│
└── 📄 README.md          # Documentation
🔮 Future Enhancements
🚀 Enhanced collaborative filtering with tensor factorization
🚀 Real-time interaction processing with websockets
🚀 Natural language query processing for recommendations
🚀 User clustering for improved similar user detection

<div align="center"> <h3>🌟 Developed for improved travel recommendations and roadmap generation 🌟</h3> </div>
