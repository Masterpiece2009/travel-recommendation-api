<h1 align="center" style="font-size:3em;font-weight:900;"> <b>Explore - Travel Recommendation API</b> </h1>
System Overview
Our sophisticated travel recommendation system provides personalized travel suggestions and itineraries using advanced algorithms combining collaborative filtering, content-based filtering, and discovery mechanisms. The system includes multiple fallback strategies ensuring robust recommendations even when primary methods fail.

Core Features
Multi-Strategy Recommendation Engine
Collaborative filtering (40% weight) - Based on similar users' interactions
Content-based filtering (60% weight) - Based on user's historical preferences
Category Matching (60%) - Matches places based on preferred categories/tags
Semantic Search (40%) - Uses NLP to find places that match interests semantically
Discovery mechanisms - Prevents recommendation bubbles with diverse suggestions
Advanced Caching System
Two-level NLP caching - Reduces processing time from 12 seconds
TTL-based cache expiration - Ensures fresh content (6-hour TTL)
Stores 7 recommendation responses per user
Background generation with locking mechanisms
Progressive Pagination
First request: 10 new places
Second request: 10 new + 10 previous (20 total)
Third+ request: 10 new + 20 previous (30 total)
Personalized Roadmap Generation
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
Key Functions
Recommendation Generation
get_candidate_places() - Retrieves candidate places matching user preferences
generate_final_recommendations() - Combines collaborative and content-based filtering with fallbacks
get_collaborative_recommendations() - Provides recommendations based on similar users' interactions
Roadmap Generation
generate_hybrid_roadmap() - Creates comprehensive travel itineraries with fallback mechanisms
get_roadmap_with_caching() - Retrieves or generates roadmaps with intelligent caching
Helper Functions
load_spacy_model() - Loads spaCy model with word vectors for semantic similarity
compute_text_similarity() - Calculates similarity between text strings
check_accessibility_compatibility() - Verifies place meets accessibility requirements
calculate_budget_compatibility() - Computes compatibility score between budgets
System Flow
Recommendation Flow

User Request → Check Cache → Generate Recommendations:
  ├── Collaborative Recommendations (40%)
  └── Content-Based Recommendations (60%)

Roadmap Generation Flow

User Request → Generate Hybrid Roadmap:
  ├── Critical Filters → Accessibility + Destination
  ├── Weighted Scoring → Budget + Group + Season
  └── Fallbacks → Nearby → Trending → Top-Rated

Recent Optimizations
✅ Modified collaborative recommendations - Returns place IDs for compatibility
✅ Implemented two-level NLP caching - Reduced processing bottleneck
✅ Enhanced similar users caching - 12-hour expiration
✅ Improved datetime handling - Fixed timezone comparison issues
✅ Implemented lazy-loading - Reduced database queries
Database Collections & Expiration
Core Data: users, places, interactions (permanent storage)
Cache Collections:
recommendations_cache (6h TTL)
roadmaps (24h TTL)
cache_locks (10m TTL)
shown_places (6h TTL)
user_keywords_cache (24h TTL)
similar_users_cache (12h TTL)
Project Structure
├── main.py            # Main application with full functionality
├── main_fixed.py      # Simplified version for deployment
├── requirements.txt   # Dependencies including geopy and spaCy
├── runtime.txt        # Specifies Python 3.10
└── README.md          # Documentation

Future Enhancements
🚀 Enhanced collaborative filtering with tensor factorization
🚀 Real-time interaction processing with websockets
🚀 Natural language query processing for recommendations
🚀 User clustering for improved similar user detection
🌟 Developed for improved travel recommendations and roadmap generation 🌟
