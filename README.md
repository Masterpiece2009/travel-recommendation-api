# Explore - Travel Recommendation API

Travel Recommendation API
<p align="center"> <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI"/> <img src="https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white" alt="MongoDB"/> <img src="https://img.shields.io/badge/Railway-0B0D0E?style=for-the-badge&logo=railway&logoColor=white" alt="Railway"/> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/> </p>
📋 Overview
This API provides sophisticated travel recommendations and roadmap generation using NLP and hybrid filtering algorithms.

🧠 Core Algorithms
1️⃣ Recommendation Algorithm
The recommendation system uses a hybrid approach that combines multiple scoring methods:

Component	Weight	Description
Category Matching	60%	Matches places based on user's preferred categories and tags
Semantic Search	40%	Uses NLP to find places that match user interests semantically
Rating Score	40%	Considers the rating of places from user reviews
Likes Score	30%	Factors in popularity based on likes
User Interactions	30%	Personalizes recommendations based on past interactions
Caching Strategy
⏱️ 6-hour TTL for cached recommendations
🔄 Stores 7 recommendation responses per user
🔍 Ensures new recommendations include both new places and places from recent requests
🔧 Uses background tasks to generate cache entries asynchronously
Key function: generate_final_recommendations()

2️⃣ Roadmap Generation Algorithm
The roadmap generation uses a hybrid filtering approach with two phases:

Phase 1: Critical Filtering (Hard Constraints)
📍 Location-based filtering to match user's preferred destinations
✅ Ensures basic compatibility with user's core requirements
Phase 2: Soft Constraint Scoring
Constraint	Weight	Description
Budget compatibility	30%	Matches places to user's budget level
Accessibility needs	20%	Ensures places meet accessibility requirements
Group type suitability	30%	Optimizes for family, solo, couples, etc.
Seasonal relevance	20%	Considers time of year for recommendations
Intelligent Caching
🔄 Only regenerates roadmaps when user preferences change
🔑 Uses a hash of user preferences to determine if regeneration is needed
🗺️ Includes geographical distance calculations for optimizing travel routes
Key function: generate_hybrid_roadmap()

🚀 API Features
🔐 Secure MongoDB connection with environment variable configuration
⚡ FastAPI endpoints with background task processing
🔄 Fallback mechanisms for NLP functionality when spaCy models lack word vectors
⏱️ Efficient caching strategies with TTL indices
🌍 Geospatial awareness for location-based recommendations
📦 Deployment
The API is deployed on Railway with proper database connection and error handling.

📂 Project Structure
│
├── 📄 main.py            # Main application file with full functionality
├── 📄 main_fixed.py      # Simplified version for deployment
├── 📄 requirements.txt   # Dependencies including geopy
├── 📄 runtime.txt        # Specifies Python 3.10
└── 📄 README.md          # Documentation
<p align="center"> <b>Developed for improved travel recommendations and roadmap generation</b><br> <small>© 2025 Travel AI Team</small> </p>
Future Enhancements
Integration with more external travel data sources
Enhanced machine learning for better place matching
Real-time collaborative filtering
Expanded user preference modeling
