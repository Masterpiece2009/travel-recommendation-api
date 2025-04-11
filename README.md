# Explore - Travel Recommendation API

## Overview
Explore is a sophisticated travel recommendation API that delivers personalized place recommendations to users. Built with FastAPI, MongoDB, and natural language processing technologies, Explore provides tailored travel suggestions based on user preferences, search history, collaborative filtering, and trending analysis.

## Features

### Core Functionality
- **Personalized Recommendations**: Combines multiple factors to create customized travel suggestions
- **Smart Caching**: Pre-computes and stores recommendation sets for faster response times
- **Multi-factor Ranking**: Uses category matching, semantic search, ratings, likes, and user interactions
- **Seamless Integration**: Simple RESTful API endpoints for easy integration with mobile and web applications

### Recommendation Algorithm
Our recommendation engine uses a weighted approach combining:
- **Category Matching (60%)**: Aligns recommendations with user's preferred travel categories
- **Semantic Search (40%)**: Uses NLP to understand the meaning behind user preferences
- **Rating Factors (40%)**: Incorporates place ratings for quality assurance
- **Popularity Metrics (30%)**: Considers likes and user interactions for trending places

### Caching Strategy
- **Performance Optimization**: Maintains 7 pre-computed recommendation sets per user
- **TTL Management**: Cache entries expire after 6 hours to ensure freshness
- **Automatic Replenishment**: Background processes monitor and refresh the cache
- **Intelligent Composition**: New recommendations include fresh places followed by relevant places from recent requests

## API Endpoints

### Recommendation Endpoints
- `GET /recommendations/{user_id}`: Retrieve personalized travel recommendations
- `GET /recommendations/{user_id}/refresh`: Force refresh recommendations cache
- `POST /recommendations/feedback`: Submit user feedback on recommendations

### Cache Management
- `GET /cache/status/{user_id}`: Check cache status for a specific user
- `GET /cache/stats`: View overall cache statistics
- `POST /cache/clear/{user_id}`: Clear cache for a specific user

### System Health
- `GET /health`: Check API health status
- `GET /metrics`: Get system performance metrics

## Technical Architecture

### Backend
- **FastAPI**: High-performance Python web framework
- **MongoDB**: NoSQL database for storing user data, places, and cache
- **NLP Processing**: Text analysis for semantic matching of places to preferences
- **Async Processing**: Background tasks for cache maintenance

### Deployment
- **Railway Platform**: Cloud hosting with automatic scaling
- **Environment Configuration**: Uses environment variables for flexible deployment
- **Reliable Operations**: Deployed with safeguards and fallbacks

## Getting Started

### Prerequisites
- Python 3.10
- MongoDB instance
- API credentials (for production use)

### Installation
1. Clone the repository
```bash
git clone https://github.com/yourusername/explore-api.git

Install dependencies
pip install -r requirements.txt
Set up environment variables
MONGO_URI=your_mongo_connection_string
MONGO_PASSWORD=your_mongo_password
Run the application
python main_fixed.py
Example Request
curl -X GET "https://api.exploretravel.com/recommendations/user123"
Example Response
{
  "recommendations": [
    {
      "place_id": "p12345",
      "name": "Eiffel Tower",
      "location": "Paris, France",
      "category": "Landmarks",
      "rating": 4.7,
      "likes": 15420,
      "description": "Iconic iron tower in Paris offering city views"
    }
  ],
  "timestamp": "2025-04-11T05:45:32.123Z",
  "cache_sequence": 15
}
Performance Metrics
Response Time: Average <100ms with cache hits
Cache Hit Rate: >90% for active users
Recommendation Relevance: 85% user satisfaction rate
Future Enhancements
Integration with more external travel data sources
Enhanced machine learning for better place matching
Real-time collaborative filtering
Expanded user preference modeling
