Explore - Travel Recommendation API
<div align="center"> <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI"/> <img src="https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white" alt="MongoDB"/> <img src="https://img.shields.io/badge/Railway-0B0D0E?style=for-the-badge&logo=railway&logoColor=white" alt="Railway"/> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/> </div> <hr style="height:1px;border:none;color:#333;background-color:#333;" />
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
<hr style="height:1px;border-width:0;color:gray;background-color:gray"> <div align="center"> <h3>🌟 Developed for improved travel recommendations and roadmap generation 🌟</h3> <p>© 2025 Travel AI Team</p> </div>
make irt more formatted ..there are texts in teh sma eline

1 minute ago
AA

abdelrahmanaly3
Explore - Travel Recommendation API
<div align="center"> <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI"/> <img src="https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white" alt="MongoDB"/> <img src="https://img.shields.io/badge/Railway-0B0D0E?style=for-the-badge&logo=railway&logoColor=white" alt="Railway"/> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/> </div> <hr>
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
<hr> <div align="center"> <h3>🌟 Developed for improved travel recommendations and roadmap generation 🌟</h3> <p>© 2025 Travel AI Team</p> </div>
