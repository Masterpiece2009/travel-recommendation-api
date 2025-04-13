<h1 align="center" style="font-size:3em;font-weight:900;">🌍 <b>Explore - Travel Recommendation API</b> 🌍</h1>
📌 Overview
Explore is a sophisticated travel recommendation system that delivers personalized travel suggestions and dynamic itineraries. It integrates collaborative filtering, content-based filtering, semantic search, and discovery mechanisms to ensure high-quality and diverse travel experiences.

Fallback strategies, intelligent caching, and progressive pagination guarantee performance, reliability, and personalization even under edge cases.

⚙️ Core Features
🧠 Multi-Strategy Recommendation Engine
Collaborative Filtering (40%) – Based on behavior of similar users.

Content-Based Filtering (60%) – Based on user’s own historical preferences.

Category Matching (60%) – Matches based on preferred categories/tags.

Semantic Search (40%) – NLP-based semantic matching to user interests.

Discovery Mechanism – Ensures recommendations remain fresh and diverse.

🚀 Advanced Caching System
Two-Level NLP Caching – Reduces NLP response time from 12s.

TTL-based Expiration – Keeps content fresh (6-hour TTL).

User-specific Cache – Stores 7 recommendation responses per user.

Locking Mechanisms – For safe background generation.

📖 Progressive Pagination
1st Request: 10 new places

2nd Request: 10 new + 10 previous (20 total)

3rd+ Request: 10 new + 20 previous (30 total)

🗺️ Personalized Roadmap Generation
✅ Critical Filters (Hard Constraints)
Location Filtering – Based on preferred regions.

Accessibility Compatibility – Filters places based on user needs.

🎯 Soft Constraint Scoring
Budget Compatibility (30%)

Accessibility Features (20%)

Group Type Suitability (30%) – Family, solo, couples, etc.

Seasonal Relevance (20%)

🧠 Intelligent Caching
Roadmaps only regenerate when user preferences change

Preference hash ensures efficient cache validation

🧩 Key Functions
🔁 Recommendation Generation
get_candidate_places() – Filters initial candidate places.

generate_final_recommendations() – Combines all strategies + fallbacks.

get_collaborative_recommendations() – User-similarity based suggestions.

📍 Roadmap Generation
generate_hybrid_roadmap() – Produces full itinerary using fallback paths.

get_roadmap_with_caching() – Returns cached or fresh roadmap.

🔧 Helper Utilities
load_spacy_model() – Loads NLP model with word vectors.

compute_text_similarity() – Measures text-based relevance.

check_accessibility_compatibility() – Verifies accessibility compliance.

calculate_budget_compatibility() – Budget scoring for suggestions.

🔄 System Flow
🧭 Recommendation Flow
scss
Copy
Edit
User Request
  └─> Check Cache
       ├── Collaborative Recommendations (40%)
       └── Content-Based Recommendations (60%)
🗺️ Roadmap Generation Flow
pgsql
Copy
Edit
User Request
  └─> Generate Hybrid Roadmap
       ├── Critical Filters → Location + Accessibility
       ├── Soft Scoring → Budget + Group Type + Season
       └── Fallbacks → Nearby → Trending → Top-Rated
🔧 Recent Optimizations
✅ Collaborative filtering returns place IDs for better performance
✅ Two-level NLP caching implemented
✅ Similar users caching (12h expiration)
✅ Timezone-aware datetime comparison
✅ Lazy-loading added to reduce DB queries

🗃️ Database Collections
📦 Core Collections (Permanent):
users

places

interactions

🧊 Cache Collections (With TTL):
Collection	TTL
recommendations_cache	6 hours
roadmaps	24 hours
cache_locks	10 minutes
shown_places	6 hours
user_keywords_cache	24 hours
similar_users_cache	12 hours
🗂️ Project Structure
bash
Copy
Edit
├── main.py             # Main app with full feature set
├── main_fixed.py       # Lightweight version for deployment
├── requirements.txt    # Dependencies (e.g. spaCy, geopy)
├── runtime.txt         # Python version (3.10)
└── README.md           # Project documentation
🌟 Future Enhancements
🚀 Tensor factorization for better collaborative filtering

🔌 Real-time interactions via WebSockets

🗣️ Natural language queries for recommendations

👥 User clustering for similarity detection

✨ Built to make discovering new travel destinations smarter, faster, and more personal.

