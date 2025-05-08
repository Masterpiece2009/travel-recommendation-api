import os
import json
import logging
import pymongo
import urllib.parse
import spacy
import math
import random
import asyncio
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic
import re
import hashlib
from langdetect import detect
from deep_translator import GoogleTranslator
import copy

def is_likely_english(text):
    """
    Helper function to determine if text is likely English based on character patterns
    and common word presence.
    
    Args:
        text: Text to analyze
        
    Returns:
        Boolean indicating if text is likely English
    """
    if not text:
        return True
        
    # Normalize the text
    normalized = text.lower().strip()
    
    # Check for English-only characters
    if not re.match(r'^[a-zA-Z\s\'\-,.!?0-9]+$', normalized):
        return False
        
    # Travel-specific vocabulary often misdetected in short queries
    travel_specific_words = {"sun", "sea", "sky", "bay", "sand", "view", "pool", "golf", "spa", "tour",
        "surf", "dive", "hike", "bike", "food", "wine", "beer", "dine", "shop", "art",
        "cafe", "lake", "peak", "hill", "path", "park", "camp", "fish", "boat", "sail",
        "walk", "trek", "swim", "reef", "cave", "rock", "snow", "ride", "trip", "rest",
        "site", "town", "city", "farm", "fort", "port", "dock", "ship", "star", "hot",
        "cold", "warm", "cool", "fair", "show", "play", "game", "room", "stay", "bed",
        "meal", "plan", "cost", "time", "date", "week", "day", "fun", "joy", "rest",
        "beach", "shore", "coast", "ocean", "island", "river", "bridge", "road", "trail"}
        
    # For very short texts (1-3 words), check against common English words
    if len(normalized.split()) <= 3:
        # This list contains very common English words that might appear in short phrases
        very_common_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
                            "i", "you", "he", "she", "it", "we", "they", "my", "your", "his", 
                            "her", "its", "our", "their", "to", "for", "with", "at", "from", 
                            "by", "on", "in", "out", "up", "down", "this", "that", "these", 
                            "those", "here", "there", "who", "what", "when", "where", "why", 
                            "how", "which", "if", "then", "than", "yes", "no", "not", "can", 
                            "will", "do", "does", "did", "have", "has", "had", "go", "went", 
                            "gone", "come", "came", "get", "got", "make", "made", "see", "saw", 
                            "say", "said", "good", "bad", "new", "old", "big", "small", "high", 
                            "low", "many", "few", "some", "any", "all", "none", "every", "each"}
        
        # Check if any of the words are common English words
        words = normalized.split()
        for word in words:
            # Remove punctuation for comparison
            clean_word = word.strip(".,!?;:'\"")
            if clean_word in very_common_words or clean_word in travel_specific_words:
                return True
            
            # For single words that are very short (≤4 letters), assume English to avoid misdetection
            if len(words) == 1 and len(clean_word) <= 4:
                return True
    
    return False

def detect_language(text):
    """
    Detect the language of a text string with improved detection for common words.
    
    Args:
        text: Text string to analyze
        
    Returns:
        Language code (e.g., 'en', 'ar')
    """
    try:
        from langdetect import detect, LangDetectException
        import re
        
        if not text or len(text.strip()) == 0:
            return "en"
        
        # Define english_pattern at the beginning to avoid reference before assignment
        english_pattern = re.compile(r'^[a-zA-Z\s\'\-,.!?]+$')
        
        # Normalize text for comparison
        normalized_text = text.lower().strip()
        
        # Check if text is likely English
        if is_likely_english(normalized_text):
            return "en"
        
        # Extensive list of common English words that might be confused with other languages
        common_english_words = {
            # Greetings and polite phrases
            "welcome", "hello", "hi", "hey", "goodbye", "bye", "thanks", "thank you", 
            "please", "sorry", "excuse me", "pardon", "cheers", "congratulations",
            "morning", "evening", "night", "afternoon", "day", "good morning", 
            "good evening", "good night", "good afternoon", "good day",
            
            # Common short words
            "yes", "no", "maybe", "ok", "okay", "fine", "sure", "of course",
            "the", "and", "or", "but", "if", "then", "than", "so", "thus",
            "a", "an", "to", "in", "on", "at", "by", "for", "with", "about",
            "from", "up", "down", "over", "under", "again", "once", "here", "there",
            
            # Time-related
            "today", "tomorrow", "yesterday", "now", "later", "soon", "never",
            "always", "often", "seldom", "sometimes", "rarely", "weekly", "daily",
            "monthly", "yearly", "minute", "hour", "second", "week", "month", "year",
            
            # Common adjectives
            "good", "bad", "nice", "great", "awesome", "cool", "hot", "cold",
            "big", "small", "large", "tiny", "huge", "little", "long", "short",
            "high", "low", "new", "old", "young", "easy", "hard", "difficult",
            "happy", "sad", "angry", "excited", "tired", "busy", "free", "cheap",
            "expensive", "beautiful", "pretty", "handsome", "ugly", "clean", "dirty",
            
            # Common verbs
            "go", "come", "get", "take", "make", "do", "have", "be", "is", "are",
            "was", "were", "has", "had", "can", "could", "will", "would", "should",
            "may", "might", "must", "need", "want", "like", "love", "hate", "see",
            "look", "watch", "hear", "listen", "speak", "say", "tell", "eat", "drink",
            "sleep", "work", "play", "walk", "run", "stop", "start", "finish",
            
            # Common nouns
            "man", "woman", "child", "boy", "girl", "person", "people", "family",
            "friend", "house", "home", "car", "bus", "train", "plane", "boat",
            "food", "water", "book", "phone", "computer", "internet", "world",
            "country", "city", "town", "street", "road", "way", "thing", "name",
            
            # Question words
            "who", "what", "where", "when", "why", "how", "which", "whose",
            
            # Numbers and quantities
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "first", "second", "third", "last", "many", "much", "more", "less", "few",
            "some", "any", "all", "none", "every", "each", "most", "least",
            
            # Travel specific (for your app)
            "travel", "hotel", "flight", "booking", "reservation", "destination",
            "trip", "journey", "vacation", "holiday", "airport", "beach", "mountain",
            "tourist", "guide", "tour", "visit", "passport", "visa", "luggage",
            "suitcase", "backpack", "map", "location", "address", "restaurant",
            "cafe", "museum", "park", "garden", "castle", "palace", "monument",
            "souvenir", "photo", "picture", "camera", "sunset", "sunrise",
            # Add problem words explicitly
            "sun", "shore", "sea", "ocean", "river", "lake", "island", "coast"
        }
        
        # Check if the text is a common English word
        if normalized_text in common_english_words:
            return "en"
        
        # Check for Arabic script characters
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
        if arabic_pattern.search(text):
            # For short texts with Arabic characters, we'll default to Arabic
            if len(text) < 20:
                return "ar"
        
        # For very short texts, try to be more precise
        if len(normalized_text) < 10:
            # These short text detection rules help avoid common misdetections
            # English specific patterns (most English words only use these characters)
            if english_pattern.match(normalized_text):
                # Check if text contains common English word patterns
                english_word_pattern = re.compile(r'\b(the|and|to|of|in|is|it|that|for|on|with|as|at|by|this|from)\b', re.IGNORECASE)
                if english_word_pattern.search(normalized_text):
                    return "en"
        
        # Use standard detection for longer texts
        try:
            detected = detect(text)
            
            # Fix common misidentification between Persian and Arabic
            if detected == "fa" and arabic_pattern.search(text):
                return "ar"  # Override Persian detection for Arabic script
            
            # Additional overrides for common misdetections of short English phrases
            if detected in ["nl", "af", "no", "da", "sv", "de", "fr"] and len(normalized_text) < 15:
                # These languages often get confused with English for short phrases
                # Do a second pass check with more context
                if english_pattern.match(normalized_text):
                    # If it looks like English orthographically and is short, trust that more
                    return "en"
                
            return detected
        except LangDetectException:
            # If langdetect fails on short text, default to English for Latin script
            if re.match(r'^[a-zA-Z\s\'\-,.!?]+$', text):
                return "en"
            raise  # Re-raise for other exceptions
            
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return "en"  # Default to English if detection fails
def translate_from_english(text, target_lang):
    """
    Translate text from English to target language.
    
    Args:
        text: English text to translate
        target_lang: Target language code
        
    Returns:
        Translated text or original if translation fails
    """
    try:
        if not text or len(text.strip()) == 0 or target_lang == 'en' or target_lang == 'und':
            return text
            
        from deep_translator import GoogleTranslator
        import hashlib
        from datetime import datetime
        
        # Create a consistent hash for cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_key = f"translate_from_en_{target_lang}_{text_hash}"
        
        # Check if we have translation in cache
        cache_result = translation_cache.find_one({"key": cache_key})
        
        if cache_result:
            logger.info(f"Using cached translation to {target_lang}")
            return cache_result["translated_text"]
            
        # Translate using Google translator
        translated = GoogleTranslator(source='en', target=target_lang).translate(text)
        logger.info(f"Translated text from English to {target_lang}")
        
        # Cache the translation result
        translation_cache.insert_one({
            "key": cache_key,
            "original_text": text,
            "translated_text": translated,
            "source_lang": 'en',
            "target_lang": target_lang,
            "timestamp": datetime.now()
        })
        
        return translated
    except Exception as e:
        logger.warning(f"Translation from English to {target_lang} failed: {e}")
        return text  # Return original if translation fails


def translate_to_english(text):
    """
    Translate non-English text to English.
    Only translates if the text is detected as non-English.
    
    Args:
        text: Text to translate
        
    Returns:
        Translated text or original if translation fails
    """
    try:
        if not text or len(text.strip()) == 0:
            return text
            
        # Detect language
        lang = detect_language(text)
        
        # Only translate if not English
        if lang != 'en':
            from deep_translator import GoogleTranslator
            
            # Check if we have translation in cache
            cache_key = f"translate_{lang}_{hash(text)}"
            cache_result = translation_cache.find_one({"key": cache_key})
            
            if cache_result:
                logger.info(f"Using cached translation from {lang}")
                return cache_result["translated_text"]
                
            # Translate using Google translator
            translated = GoogleTranslator(source=lang, target='en').translate(text)
            logger.info(f"Translated text from {lang} to English")
            
            # Cache the translation result
            translation_cache.insert_one({
                "key": cache_key,
                "original_text": text,
                "translated_text": translated,
                "source_lang": lang,
                "timestamp": datetime.now()
            })
            
            return translated
        return text
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text  # Return original if translation fails
def translate_roadmap_results(roadmap_list, target_language):
    """
    Translate roadmap results to the target language
    
    Parameters:
    - roadmap_list: List of roadmap items to translate
    - target_language: Target language code (e.g., 'ar' for Arabic)
    
    Returns:
    - Translated roadmap list
    """
    try:
        # Skip translation if no target language or if no items to translate
        if not roadmap_list or not target_language or target_language in ['en', 'und']:
            return roadmap_list
            
        logger.info(f"Translating roadmap results to {target_language}")
        translated_results = []
        
        # Dictionary of landmark types in different languages
        landmark_types = {
            "fr": {
                "Château": "castle/palace",
                "Palais": "palace",
                "Cathédrale": "cathedral",
                "Musée": "museum",
                "Basilique": "basilica",
                "Église": "church",
                "Tour": "tower",
                "Pont": "bridge"
            },
            "es": {
                "Palacio": "palace",
                "Catedral": "cathedral",
                "Museo": "museum",
                "Iglesia": "church",
                "Puente": "bridge"
            },
            "it": {
                "Palazzo": "palace",
                "Cattedrale": "cathedral",
                "Museo": "museum",
                "Chiesa": "church",
                "Ponte": "bridge"
            },
            "de": {
                "Schloss": "castle",
                "Palast": "palace",
                "Dom": "cathedral",
                "Museum": "museum",
                "Kirche": "church",
                "Brücke": "bridge"
            }
        }
        
        for item in roadmap_list:
            # Deep copy to avoid modifying the original
            translated_item = copy.deepcopy(item)
            
            # Translate place details
            if "place" in translated_item:
                place = translated_item["place"]
                
                # Improved place name translation
                if "name" in place and isinstance(place["name"], str):
                    name = place["name"]
                    name_language = detect_language(name)
                    
                    # If already in target language, skip translation
                    if name_language == target_language:
                        logger.info(f"Name already in target language ({name_language}): {name}")
                        continue
                    
                    # Check if it's a landmark name in a known language
                    is_landmark = False
                    english_landmark_type = None
                    proper_name = None
                    
                    # Identify if this is a landmark and get its type
                    if name_language in landmark_types:
                        for landmark in landmark_types[name_language]:
                            if name.startswith(landmark):
                                is_landmark = True
                                english_landmark_type = landmark_types[name_language][landmark]
                                
                                # Extract the proper name portion
                                proper_name = name[len(landmark):].strip()
                                
                                # Handle common prepositions
                                if name_language == "fr" and proper_name.startswith("de "):
                                    proper_name = proper_name[3:].strip()
                                elif name_language == "fr" and proper_name.startswith("du "):
                                    proper_name = proper_name[3:].strip()
                                elif name_language == "es" and proper_name.startswith("de "):
                                    proper_name = proper_name[3:].strip()
                                elif name_language == "it" and proper_name.startswith("di "):
                                    proper_name = proper_name[3:].strip()
                                elif name_language == "de" and proper_name.startswith("von "):
                                    proper_name = proper_name[4:].strip()
                                
                                break
                    
                    # Translation strategy based on name type
                    if is_landmark:
                        # Strategy 1: Try direct translation from source language
                        try:
                            from deep_translator import GoogleTranslator
                            direct_translated = GoogleTranslator(source=name_language, target=target_language).translate(name)
                            
                            # If direct translation succeeded and changed the text
                            if direct_translated != name:
                                place["name"] = direct_translated
                                logger.info(f"Directly translated landmark name: {name} -> {place['name']}")
                            else:
                                # Strategy 2: Translate landmark type and proper name separately
                                translated_type = translate_from_english(english_landmark_type, target_language)
                                translated_proper_name = None
                                
                                # Try to preserve proper names unless they have meaning
                                if proper_name:
                                    proper_name_language = detect_language(proper_name)
                                    # If proper name is detectable as a language, translate it
                                    if proper_name_language != "und":
                                        if proper_name_language == "en":
                                            translated_proper_name = translate_from_english(proper_name, target_language)
                                        else:
                                            english_proper_name = translate_to_english(proper_name)
                                            translated_proper_name = translate_from_english(english_proper_name, target_language)
                                    else:
                                        # Keep proper names as is
                                        translated_proper_name = proper_name
                                
                                # Combine translated parts
                                if translated_proper_name:
                                    place["name"] = f"{translated_type} {translated_proper_name}"
                                else:
                                    place["name"] = translated_type
                                
                                logger.info(f"Translated landmark name by parts: {name} -> {place['name']}")
                        except Exception as e:
                            logger.warning(f"Direct landmark translation failed: {e}, falling back to standard method")
                            # Fall back to standard method
                            if name_language == "en":
                                place["name"] = translate_from_english(name, target_language)
                            else:
                                english_name = translate_to_english(name)
                                place["name"] = translate_from_english(english_name, target_language)
                    else:
                        # Standard translation path for regular place names
                        if name_language == "en" or name_language == "und":
                            place["name"] = translate_from_english(name, target_language)
                            logger.info(f"Translated name from English: {name} -> {place['name']}")
                        else:
                            # Try direct translation first for known languages
                            try_direct = False
                            
                            # Some language pairs work better with direct translation
                            if name_language in ["fr", "es", "de", "it", "ja", "zh", "ru", "ar"]:
                                try_direct = True
                            
                            if try_direct:
                                try:
                                    from deep_translator import GoogleTranslator
                                    direct_translated = GoogleTranslator(source=name_language, target=target_language).translate(name)
                                    if direct_translated != name:
                                        place["name"] = direct_translated
                                        logger.info(f"Directly translated name: {name} -> {place['name']}")
                                    else:
                                        # Fall back to English as intermediary
                                        english_name = translate_to_english(name)
                                        place["name"] = translate_from_english(english_name, target_language)
                                        logger.info(f"Translated name via English: {name} ({name_language}) -> {english_name} -> {place['name']}")
                                except Exception:
                                    # Fall back to English as intermediary
                                    english_name = translate_to_english(name)
                                    place["name"] = translate_from_english(english_name, target_language)
                                    logger.info(f"Translated name via English: {name} ({name_language}) -> {english_name} -> {place['name']}")
                            else:
                                # Standard path via English
                                english_name = translate_to_english(name)
                                place["name"] = translate_from_english(english_name, target_language)
                                logger.info(f"Translated name via English: {name} ({name_language}) -> {english_name} -> {place['name']}")
                
                # Rest of the translation function remains unchanged
                # Improve description translation with language detection
                if "description" in place and isinstance(place["description"], str):
                    description = place["description"]
                    description_language = detect_language(description)
                    
                    if description_language == "en" or description_language == "und":
                        place["description"] = translate_from_english(description, target_language)
                    elif description_language != target_language:
                        english_description = translate_to_english(description)
                        place["description"] = translate_from_english(english_description, target_language)
                
                # Improve city translation with language detection
                if "location" in place and "city" in place["location"] and isinstance(place["location"]["city"], str):
                    city = place["location"]["city"]
                    city_language = detect_language(city)
                    
                    if city_language == "en" or city_language == "und":
                        place["location"]["city"] = translate_from_english(city, target_language)
                    elif city_language != target_language:
                        english_city = translate_to_english(city)
                        place["location"]["city"] = translate_from_english(english_city, target_language)
                
                # Improve country translation with language detection
                if "location" in place and "country" in place["location"] and isinstance(place["location"]["country"], str):
                    country = place["location"]["country"]
                    country_language = detect_language(country)
                    
                    if country_language == "en" or country_language == "und":
                        place["location"]["country"] = translate_from_english(country, target_language)
                    elif country_language != target_language:
                        english_country = translate_to_english(country)
                        place["location"]["country"] = translate_from_english(english_country, target_language)
                
                # Improve tags translation with language detection
                if "tags" in place and isinstance(place["tags"], list):
                    translated_tags = []
                    for tag in place["tags"]:
                        if isinstance(tag, str):
                            tag_language = detect_language(tag)
                            
                            if tag_language == "en" or tag_language == "und":
                                translated_tag = translate_from_english(tag, target_language)
                            elif tag_language != target_language:
                                english_tag = translate_to_english(tag)
                                translated_tag = translate_from_english(english_tag, target_language)
                            else:
                                translated_tag = tag
                                
                            translated_tags.append(translated_tag)
                        else:
                            translated_tags.append(tag)
                    place["tags"] = translated_tags
                
                # Improve category translation with language detection
                if "category" in place and isinstance(place["category"], str):
                    category = place["category"]
                    category_language = detect_language(category)
                    
                    if category_language == "en" or category_language == "und":
                        place["category"] = translate_from_english(category, target_language)
                    elif category_language != target_language:
                        english_category = translate_to_english(category)
                        place["category"] = translate_from_english(english_category, target_language)
                
                # Improve accessibility features translation with language detection
                if "accessibility" in place and isinstance(place["accessibility"], list):
                    translated_accessibility = []
                    for feature in place["accessibility"]:
                        if isinstance(feature, str):
                            feature_language = detect_language(feature)
                            
                            if feature_language == "en" or feature_language == "und":
                                translated_feature = translate_from_english(feature, target_language)
                            elif feature_language != target_language:
                                english_feature = translate_to_english(feature)
                                translated_feature = translate_from_english(english_feature, target_language)
                            else:
                                translated_feature = feature
                                
                            translated_accessibility.append(translated_feature)
                        else:
                            translated_accessibility.append(feature)
                    place["accessibility"] = translated_accessibility
                
                # Improve appropriate_time months translation with language detection
                if "appropriate_time" in place and isinstance(place["appropriate_time"], list):
                    translated_months = []
                    for month in place["appropriate_time"]:
                        if isinstance(month, str):
                            month_language = detect_language(month)
                            
                            if month_language == "en" or month_language == "und":
                                translated_month = translate_from_english(month, target_language)
                            elif month_language != target_language:
                                english_month = translate_to_english(month)
                                translated_month = translate_from_english(english_month, target_language)
                            else:
                                translated_month = month
                                
                            translated_months.append(translated_month)
                        else:
                            translated_months.append(month)
                    place["appropriate_time"] = translated_months
            
            # Improve next destination translation with language detection
            if "next_destination" in translated_item and isinstance(translated_item["next_destination"], str):
                next_destination = translated_item["next_destination"]
                next_destination_language = detect_language(next_destination)
                
                if next_destination_language == "en" or next_destination_language == "und":
                    translated_item["next_destination"] = translate_from_english(next_destination, target_language)
                elif next_destination_language != target_language:
                    english_next_destination = translate_to_english(next_destination)
                    translated_item["next_destination"] = translate_from_english(english_next_destination, target_language)
            
            translated_results.append(translated_item)
        
        logger.info(f"Translated {len(translated_results)} roadmap items to {target_language}")
        return translated_results
    except Exception as e:
        logger.error(f"Error translating roadmap results: {str(e)}")
        return roadmap_list  # Return original if translation fails
# Define DummyNLP in global scope for fallback
class DummyNLP:
    def __init__(self):
        self.name = "DummyNLP-Fallback"
        self.vocab = type('obj', (object,), {
            'vectors': type('obj', (object,), {'n_keys': 0})
        })
        
    def __call__(self, text):
        class DummyDoc:
            def __init__(self, text):
                self.text = text
                self.vector = [0] * 300  # Empty vector
                self.vector_norm = 0
                
            def similarity(self, other):
                # Fallback similarity using Jaccard index on word overlap
                words1 = set(self.text.lower().split())
                words2 = set(other.text.lower().split())
                
                if not words1 or not words2:
                    return 0
                    
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                
                return len(intersection) / len(union)
        
        return DummyDoc(text)

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Add this AFTER your logger is defined
try:
    import langdetect
    import deep_translator
    print("✅ Language detection and translation packages loaded successfully")
    logger.info("✅ Language detection and translation packages loaded successfully")
except ImportError as e:
    print(f"❌ Error loading language packages: {e}")
    logger.error(f"❌ Error loading language packages: {e}")
# --- Task Priority System ---
class TaskPriority:
    """Priority levels for background tasks"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TaskManager:
    """Simple task manager with priority handling"""
    def __init__(self):
        # Get logger for this class
        self.logger = logging.getLogger(__name__)
        
        # Track concurrent tasks by priority
        self.running_tasks = {
            TaskPriority.HIGH: 0,
            TaskPriority.MEDIUM: 0,
            TaskPriority.LOW: 0
        }
        
        # Define limits for each priority
        self.limits = {
            TaskPriority.HIGH: 5,   # Allow 5 high priority tasks
            TaskPriority.MEDIUM: 3, # Allow 3 medium priority tasks
            TaskPriority.LOW: 2     # Allow 2 low priority tasks
        }
        
        # Create asyncio semaphores for concurrency control
        self.semaphores = {}
        for priority, limit in self.limits.items():
            self.semaphores[priority] = asyncio.Semaphore(limit)
        
        self.logger.info(f"Task manager initialized with limits: {self.limits}")
    
    async def run_task(self, priority, func, *args, **kwargs):
        """Run a task with specified priority"""
        # Default to LOW if invalid priority
        if priority not in self.semaphores:
            self.logger.warning(f"Invalid priority: {priority}, using LOW")
            priority = TaskPriority.LOW
        
        # Use semaphore to limit concurrent execution
        async with self.semaphores[priority]:
            start_time = datetime.now()
            self.running_tasks[priority] += 1
            self.logger.info(f"Starting {priority} task: {func.__name__} (running: {self.running_tasks})")
            
            try:
                # Run function based on whether it's async or not
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    # Run non-async functions in thread pool
                    result = await asyncio.to_thread(func, *args, **kwargs)
                
                duration = (datetime.now() - start_time).total_seconds()
                self.logger.info(f"Completed {priority} task: {func.__name__} in {duration:.2f}s")
                return result
            except Exception as e:
                self.logger.error(f"Error in {priority} task {func.__name__}: {str(e)}")
                raise
            finally:
                self.running_tasks[priority] -= 1
    
    def schedule_task(self, background_tasks, priority, func, *args, **kwargs):
        """Schedule a task to run in the background with priority"""
        async def wrapped_task():
            await self.run_task(priority, func, *args, **kwargs)
        
        background_tasks.add_task(wrapped_task)
        self.logger.info(f"Scheduled {priority} background task: {func.__name__}")

# Create a global instance
task_manager = TaskManager()
# Import MongoClient at the top of your file
from pymongo import MongoClient

# ... (other imports)

# Define connect_mongo function
def connect_mongo(uri, retries=3):
    """Connect to MongoDB with retry logic"""
    last_error = None
    
    for attempt in range(retries):
        try:
            client = MongoClient(uri)
            # Test the connection
            client.admin.command('ping')
            print(f"✅ MongoDB connection successful (attempt {attempt + 1})")
            return client
        except Exception as e:
            last_error = e
            print(f"❌ MongoDB connection attempt {attempt + 1} failed: {e}")
            
    # If we get here, all retries failed
    raise Exception(f"❌ MongoDB connection failed after {retries} attempts: {last_error}")

# Securely Connect to MongoDB
password = os.environ.get("MONGO_PASSWORD", "cmCqBjtQCQDWbvlo")  # Fallback for development
encoded_password = urllib.parse.quote_plus(password)

MONGO_URI = f"mongodb+srv://shehabwww153:{encoded_password}@userauth.rvtb5.mongodb.net/travel_app?retryWrites=true&w=majority&appName=userAuth"
client = connect_mongo(MONGO_URI)
db = client["travel_app"]
# Define Collections
users_collection = db["users"]
places_collection = db["places"]
interactions_collection = db["interactions"]
search_queries_collection = db["search_queries"]
travel_preferences_collection = db["user_travel_preferences"]
recommendations_cache_collection = db["recommendations_cache"]
shown_places_collection = db["shown_places"]
roadmaps_collection = db["roadmaps"]  # For roadmaps
cache_locks_collection = db["cache_locks"]  # New collection for tracking locks
# New cache collections for performance optimizations
user_keywords_cache = db["user_keywords_cache"]
keyword_similarity_cache = db["keyword_similarity_cache"] 
similar_users_cache = db["similar_users_cache"]
# Add translation cache collection
translation_cache = db["translation_cache"]
# Add reviews collection
reviews_collection = db["reviews"]  # New collection for place reviews

# Later, add this with your other TTL indexes
try:
    translation_cache.create_index(
        [("timestamp", pymongo.ASCENDING)],
        expireAfterSeconds=604800  # 7 days
    )
    logger.info("✅ Created TTL index on translation_cache collection")
except Exception as e:
    logger.error(f"❌ Error creating TTL index on translation_cache collection: {e}")
# Create TTL indexes for new collections
user_keywords_cache.create_index("timestamp", expireAfterSeconds=86400)  # 24 hours
keyword_similarity_cache.create_index("timestamp", expireAfterSeconds=86400)  # 24 hours
similar_users_cache.create_index("timestamp", expireAfterSeconds=43200)  # 12 hours

# --- CREATE TTL INDEXES ---
# These indexes automatically remove documents after a specified time period

# TTL index for roadmaps (expires after 24 hours)
try:
    roadmaps_collection.create_index(
        [("created_at", pymongo.ASCENDING)],
        expireAfterSeconds=86400  # 24 hours
    )
    logger.info("✅ Created TTL index on roadmaps collection")
except Exception as e:
    logger.error(f"❌ Error creating TTL index on roadmaps collection: {e}")

# TTL index for recommendations cache (expires after 6 hours)
try:
    recommendations_cache_collection.create_index(
        [("timestamp", pymongo.ASCENDING)],
        expireAfterSeconds=21600  # 6 hours
    )
    logger.info("✅ Created TTL index on recommendations_cache collection")
except Exception as e:
    logger.error(f"❌ Error creating TTL index on recommendations_cache collection: {e}")

try:
    # Check if the index exists first
    existing_indexes = list(shown_places_collection.list_indexes())
    ttl_index_exists = any(idx.get('name') == 'last_updated_1' for idx in existing_indexes)
    
    # If the index exists with different options, drop it first
    if ttl_index_exists:
        shown_places_collection.drop_index('last_updated_1')
        logger.info("Dropped existing TTL index on shown_places collection")
    
    # Create new index on last_updated field
    shown_places_collection.create_index(
        [("last_updated", pymongo.ASCENDING)],  # Changed from "timestamp" to "last_updated"
        expireAfterSeconds=21600  # 6 hours
    )
    logger.info("✅ Created TTL index on shown_places collection using last_updated field")
except Exception as e:
    logger.error(f"❌ Error creating TTL index on shown_places collection: {e}")
# TTL index for cache locks (expires after 10 minutes)
try:
    cache_locks_collection.create_index(
        [("timestamp", pymongo.ASCENDING)],
        expireAfterSeconds=600  # 10 minutes (safety cleanup for stale locks)
    )
    logger.info("✅ Created TTL index on cache_locks collection")
except Exception as e:
    logger.error(f"❌ Error creating TTL index on cache_locks collection: {e}")

# Create index on user_id field for better query performance
for collection_name in ["recommendations_cache", "shown_places", "roadmaps", "cache_locks"]:
    try:
        db[collection_name].create_index([("user_id", pymongo.ASCENDING)])
        logger.info(f"✅ Created user_id index on {collection_name} collection")
    except Exception as e:
        logger.error(f"❌ Error creating index on {collection_name}: {e}")

# --- Initialize spaCy model ---
def load_spacy_model(model="en_core_web_md", retries=2):  # Use medium model by default
    """Attempts to load the spaCy model with better vector checking."""
    logger.info(f"🔄 Attempting to load spaCy model: {model}")
    
    for attempt in range(retries):
        try:
            nlp = spacy.load(model)
            
            # Verify that the model has word vectors
            test_doc = nlp("travel")
            has_vectors = nlp.vocab.vectors.n_keys > 0 and test_doc.vector_norm > 0
            
            if has_vectors:
                logger.info(f"✅ Successfully loaded spaCy model: {model} WITH WORD VECTORS")
                return nlp
            else:
                logger.warning(f"⚠️ Model {model} loaded but NO WORD VECTORS found!")
                
                # If this is the 'md' model and it doesn't have vectors, try 'sm' model
                if model == "en_core_web_md" and attempt == 0:
                    logger.info("⚠️ Medium model doesn't have vectors. Attempting to download vectors...")
                    try:
                        import subprocess
                        result = subprocess.run([sys.executable, "-m", "spacy", "download", model], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            logger.info(f"✅ Successfully downloaded model: {model}")
                            continue  # Try loading again
                    except Exception as download_err:
                        logger.error(f"❌ Failed to download model: {download_err}")
                
                # If we can't fix the current model, fall back to the small model
                if model != "en_core_web_sm":
                    logger.info("🔄 Falling back to small model...")
                    return load_spacy_model("en_core_web_sm", 1)
        except Exception as e:
            logger.error(f"❌ Error loading NLP model (Attempt {attempt + 1}/{retries}): {e}")
            try:
                logger.info(f"📥 Downloading spaCy model: {model}")
                import subprocess
                result = subprocess.run([sys.executable, "-m", "spacy", "download", model], 
                                       capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"✅ Successfully downloaded model: {model}")
                else:
                    logger.error(f"❌ Failed to download model: {result.stderr}")
            except Exception as download_err:
                logger.error(f"❌ Failed to download model: {download_err}")

    # Return dummy NLP object from the global class
    logger.warning("⚠️ CRITICAL: Using dummy NLP model as fallback! Semantic search will use word overlap instead.")
    return DummyNLP()

# Try to load the model with word vectors
nlp = load_spacy_model()

# Check if model has word vectors and log clearly
test_text = "travel"
test_doc = nlp(test_text)
has_vectors = hasattr(test_doc, 'vector_norm') and test_doc.vector_norm > 0

if has_vectors:
    logger.info("✅ SUCCESS: NLP Model loaded with WORD VECTORS - semantic search will work properly")
else:
    logger.warning("⚠️ WARNING: NLP Model doesn't have word vectors - semantic search will use fallback algorithm")
# Initialize FastAPI app
app = FastAPI(
    title="Travel API",
    description="API for travel recommendations and roadmaps",
    version="2.0.0"  # Updated version
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- PART 2: MODELS AND SHARED UTILITY FUNCTIONS ---

# --- Pydantic Models ---
class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: Optional[int] = 10

class RoadmapRequest(BaseModel):
    user_id: str

class SearchRequest(BaseModel):
    user_id: str
    query: str
    limit: Optional[int] = 10

# --- Shared Utility Functions ---

def get_user_data(user_id):
    """Get complete user data including preferences"""
    user = users_collection.find_one({"_id": user_id})
    
    if not user:
        logger.warning(f"User {user_id} not found")
        return None
        
    return user

def get_user_preferences(user_id):
    """Get user general preferences (categories & tags only)"""
    user = get_user_data(user_id)
    
    if not user:
        return None
        
    # Handle the nested preferences structure
    preferences = user.get("preferences", {})
    
    return {
        "preferred_categories": preferences.get("categories", []),
        "preferred_tags": preferences.get("tags", []),
    }

def get_user_travel_preferences(user_id):
    """Get user travel-specific preferences, including budget"""
    travel_prefs = travel_preferences_collection.find_one(
        {"user_id": user_id}
    )
    
    if not travel_prefs:
        logger.warning(f"No travel preferences found for user {user_id}")
        return None
    
    # Get the group_type value and ensure it's a string
    group_type = travel_prefs.get("group_type", "")
    if isinstance(group_type, list):
        # If it's a list, join it into a single string
        group_type = ", ".join(str(item) for item in group_type)
    elif not isinstance(group_type, str):
        # If it's any other non-string type, convert to string
        group_type = str(group_type)
        
    return {
        "destinations": travel_prefs.get("destinations", []),
        "travel_dates": travel_prefs.get("travel_dates", ""),
        "accessibility_needs": travel_prefs.get("accessibility_needs", []),
        "budget": travel_prefs.get("budget", "medium"),  # Default to 'medium' if missing
        "group_type": group_type  # Now guaranteed to be a string
    }

def compute_text_similarity(text1, text2):
    """
    Compute similarity between two text strings.
    Uses spaCy word vectors if available, falls back to word overlap otherwise.
    Handles non-English text by translating to English first.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0
        
    try:
        # Check if either text needs translation
        text1_lang = detect_language(text1)
        text2_lang = detect_language(text2)
        
        # Translate if needed
        if text1_lang != 'en':
            original_text1 = text1
            text1 = translate_to_english(text1)
            logger.debug(f"Translated '{original_text1}' ({text1_lang}) to '{text1}'")
            
        if text2_lang != 'en':
            original_text2 = text2
            text2 = translate_to_english(text2)
            logger.debug(f"Translated '{original_text2}' ({text2_lang}) to '{text2}'")
        
        # Try using spaCy word vectors
        doc1 = nlp(text1.lower())
        doc2 = nlp(text2.lower())
        
        # Check if vectors are available (not all models have vectors)
        if hasattr(doc1, 'vector_norm') and doc1.vector_norm > 0 and hasattr(doc2, 'vector_norm') and doc2.vector_norm > 0:
            similarity = doc1.similarity(doc2)
            logger.debug(f"Vector similarity between texts: {similarity:.2f}")
            return similarity
        
        # Fall back to basic word overlap if vectors aren't available
        words1 = set(word.lower() for word in text1.split())
        words2 = set(word.lower() for word in text2.split())
        
        if not words1 or not words2:
            return 0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0
        logger.debug(f"Jaccard similarity between texts: {jaccard_similarity:.2f}")
        return jaccard_similarity
        
    except Exception as e:
        logger.error(f"Error computing text similarity: {str(e)}")
        
        # Emergency fallback
        return 0
def find_similar_terms(word, limit=3):
    """
    Find terms similar to the given word using word vectors.
    Supports non-English words by translating to English first.
    
    Args:
        word: Word to find similar terms for
        limit: Maximum number of similar terms to return
        
    Returns:
        List of similar terms
    """
    # Helper function to safely extract float values
    def ensure_float(value):
        if isinstance(value, dict):
            # Handle MongoDB numeric types
            if "$numberDouble" in value:
                return float(value["$numberDouble"])
            if "$numberInt" in value:
                return float(int(value["$numberInt"]))
            if "$numberLong" in value:
                return float(int(value["$numberLong"]))
            return 0.0  # Default if it's an unrecognized dict
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
            
    try:
        # Detect language and translate if needed
        original_word = word
        detected_language = detect_language(word)
        
        if detected_language != 'en':
            word = translate_to_english(word)
            logger.debug(f"Translated '{original_word}' ({detected_language}) to '{word}'")
            
        # Check if NLP is available and has vectors
        if not nlp or not hasattr(nlp, 'vocab') or not nlp.vocab.vectors.size:
            return []
        
        # Process the word with spaCy
        word_processed = nlp(word)
        
        if not word_processed or not word_processed[0].has_vector:
            return []
        
        # Get the word vector
        word_vector = word_processed[0].vector
        
        # Find similar terms from our places data
        similar_terms = []
        
        # Get a list of common terms from places
        common_terms = []
        
        # Sample up to 100 places for performance
        sample_places = list(places_collection.find().limit(100))
        
        # Extract terms from places
        for place in sample_places:
            # Extract words from name, categories, tags
            terms = place.get("name", "").split() + place.get("categories", []) + place.get("tags", [])
            common_terms.extend([term.lower() for term in terms if len(term) > 3])
        
        # Count occurrences to find most common terms
        term_counts = Counter(common_terms)
        common_terms = [term for term, count in term_counts.most_common(100)]
        
        # Calculate similarity for each term
        term_similarities = []
        
        for term in common_terms:
            term_doc = nlp(term)
            if term_doc and term_doc[0].has_vector:
                similarity = term_doc[0].similarity(word_processed[0])
                # Convert similarity to ensure it's a standard Python float
                safe_similarity = ensure_float(similarity)
                if safe_similarity > 0.4 and term != word:  # Only include sufficiently similar terms
                    term_similarities.append((term, safe_similarity))
        
        # Sort by similarity and take top results
        # Using the safe float values for comparison
        term_similarities.sort(key=lambda x: ensure_float(x[1]), reverse=True)
        similar_terms = [term for term, _ in term_similarities[:limit]]
        
        return similar_terms
        
    except Exception as e:
        logger.error(f"Error finding similar terms for '{word}': {e}")
        return []

def find_places_by_keyword_similarity(keywords, excluded_place_ids=None, limit=20, fallback_level=0):
    """
    Find places similar to the given keywords using NLP, with similarity caching.
    Supports non-English keywords by translating to English first.
    
    Args:
        keywords: List of keywords to match
        excluded_place_ids: List of place IDs to exclude
        limit: Maximum number of places to return
        fallback_level: Fallback level (0=full NLP, 1=basic NLP, 2=text search, 3=random)
        
    Returns:
        List of places matching the keywords
    """
    # Helper function to safely extract float values
    def ensure_float(value):
        if isinstance(value, dict):
            # Handle MongoDB numeric types
            if "$numberDouble" in value:
                return float(value["$numberDouble"])
            if "$numberInt" in value:
                return float(int(value["$numberInt"]))
            if "$numberLong" in value:
                return float(int(value["$numberLong"]))
            return 0.0  # Default if it's an unrecognized dict
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
            
    if not keywords:
        return []
        
    excluded_place_ids = excluded_place_ids or []
    
    # Translate non-English keywords
    translated_keywords = []
    original_keywords = keywords.copy()
    
    for keyword in keywords:
        detected_language = detect_language(keyword)
        if detected_language != 'en':
            translated = translate_to_english(keyword)
            logger.debug(f"Translated keyword '{keyword}' ({detected_language}) to '{translated}'")
            translated_keywords.append(translated)
        else:
            translated_keywords.append(keyword)
    
    # Use translated keywords for search and caching
    keywords = translated_keywords
    cache_key = "_".join(sorted(keywords))
    
    try:
        # Check if we have a cached result for these keywords
        cached_result = keyword_similarity_cache.find_one({"cache_key": cache_key})
        
        if cached_result and "place_ids" in cached_result:
            # Get the places from the database
            place_ids = cached_result["place_ids"]
            
            # Filter out excluded places
            filtered_place_ids = [pid for pid in place_ids if pid not in excluded_place_ids]
            
            # If we have enough places after filtering, use the cache
            if len(filtered_place_ids) >= min(5, limit):
                logger.info(f"Using cached keyword similarity results for {cache_key}")
                
                # Get the actual place documents
                places = list(places_collection.find({"_id": {"$in": filtered_place_ids}}))
                
                # Sort by original order (preserves similarity ranking)
                order_map = {pid: idx for idx, pid in enumerate(filtered_place_ids)}
                places.sort(key=lambda p: order_map.get(p["_id"], 999))
                
                return places[:limit]
        
        # No cache or not enough places after filtering, do the full search
        start_time = time.time()
        results = []
        
        # Try different approaches based on fallback level
        if fallback_level == 0 and nlp and hasattr(nlp, 'vocab') and nlp.vocab.vectors.size:
            # Full NLP with word vectors (most advanced)
            logger.info(f"Using full spaCy NLP with word vectors")
            
            # Get all places
            all_places = list(places_collection.find({"_id": {"$nin": excluded_place_ids}}))
            
            # Calculate similarity scores for all places
            place_scores = []
            
            for place in all_places:
                score = 0
                place_text = f"{place.get('name', '')} {place.get('description', '')} {' '.join(place.get('categories', []))} {' '.join(place.get('tags', []))}"
                
                # Calculate similarity for each keyword
                for keyword in keywords:
                    keyword_score = compute_text_similarity(keyword, place_text)
                    # Ensure the score is a standard Python float
                    keyword_score = ensure_float(keyword_score)
                    score += keyword_score
                
                # Store the normalized score as a standard Python float
                normalized_score = score / len(keywords)
                place_scores.append((place, normalized_score))
            
            # Sort by score (descending) and take top results
            # Use the ensure_float function to safely extract the score
            place_scores.sort(key=lambda x: ensure_float(x[1]), reverse=True)
            results = [place for place, score in place_scores[:limit]]
            
            # Cache the result (only place IDs to save space)
            place_ids = [place["_id"] for place in results]
            
            keyword_similarity_cache.update_one(
                {"cache_key": cache_key},
                {"$set": {
                    "cache_key": cache_key,
                    "keywords": keywords,
                    "original_keywords": original_keywords,
                    "place_ids": place_ids,
                    "fallback_level": fallback_level,
                    "timestamp": datetime.now()
                }},
                upsert=True
            )
            
        elif fallback_level <= 1 and nlp:
            # Basic NLP without vectors
            logger.info(f"Using basic spaCy NLP (fallback level: 1)")
            # Use your existing implementation for this fallback level
            
        elif fallback_level <= 2:
            # Text search fallback
            logger.info(f"Using text search (fallback level: 2)")
            # Use your existing implementation for this fallback level
            
        else:
            # Random fallback
            logger.info(f"Using random selection (fallback level: 3)")
            # Use your existing implementation for this fallback level
            
        logger.info(f"Found {len(results)} places via search keywords (fallback level: {fallback_level})")
        return results
        
    except Exception as e:
        logger.error(f"Error in find_places_by_keyword_similarity: {e}")
        # Return empty list on error
        return []
def extract_search_keywords(user_id, user_preferences=None, max_keywords=5):
    """
    Extract search keywords for a user based on their preferences, with caching.
    
    Args:
        user_id: User ID to extract keywords for
        user_preferences: Optional user preferences dict
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List of extracted keywords for this user
    """
    try:
        # Check cache first
        cached_keywords = user_keywords_cache.find_one({"user_id": user_id})
        
        if cached_keywords and "keywords" in cached_keywords:
            logger.info(f"Using cached keywords for user {user_id}")
            return cached_keywords["keywords"]
        
        # If no cache, extract keywords
        if not user_preferences:
            user_doc = users_collection.find_one({"_id": user_id})
            if not user_doc:
                logger.warning(f"User {user_id} not found when extracting keywords")
                return []
            user_preferences = user_doc
            
        # Get categories and tags from user preferences
        categories = user_preferences.get("categories", [])
        tags = user_preferences.get("tags", [])
        
        # Get all relevant words from categories and tags
        all_words = []
        
        # Process categories - translate if needed
        for category in categories:
            # Check language and translate if needed
            cat_lang = detect_language(category)
            if cat_lang != 'en':
                orig_category = category
                category = translate_to_english(category)
                logger.debug(f"Translated category '{orig_category}' ({cat_lang}) to '{category}'")
            
            all_words.extend([category] * 3)  # Higher weight for categories
            
        # Process tags - translate if needed
        for tag in tags:
            tag_lang = detect_language(tag)
            if tag_lang != 'en':
                orig_tag = tag
                tag = translate_to_english(tag)
                logger.debug(f"Translated tag '{orig_tag}' ({tag_lang}) to '{tag}'")
            
            all_words.append(tag)
        
        # Extract additional relevant terms using NLP
        if nlp:
            # Create a text representation of preferences
            preference_text = " ".join(all_words)
            
            # Process the text with spaCy
            doc = nlp(preference_text)
            
            # Extract additional keywords from similar terms in our vocabulary
            # based on vector similarity
            additional_keywords = []
            
            for token in doc:
                if token.has_vector and not token.is_stop and token.is_alpha:
                    similar_terms = find_similar_terms(token.text, 3)
                    additional_keywords.extend(similar_terms)
            
            # Add these to all_words
            all_words.extend(additional_keywords)
        
        # Count occurrences to find most common
        keyword_counts = Counter(all_words)
        
        # Get the most common keywords
        keywords = [kw for kw, _ in keyword_counts.most_common(max_keywords)]
        
        # Cache the result
        user_keywords_cache.update_one(
            {"user_id": user_id},
            {"$set": {
                "user_id": user_id,
                "keywords": keywords,
                "timestamp": datetime.now()
            }},
            upsert=True
        )
        
        return keywords
        
    except Exception as e:
        logger.error(f"Error extracting search keywords for user {user_id}: {e}")
        return []

def parse_travel_dates(travel_dates_str):
    """
    Parse the travel_dates string to extract month information.
    
    Args:
        travel_dates_str: String containing travel dates (e.g. "March 2025", "now", "August 2025")
        
    Returns:
        String containing the month name or None if not parseable
    """
    if not travel_dates_str:
        return None
        
    # If user selected "now", return current month
    if travel_dates_str.lower() == "now":
        return datetime.now().strftime("%B")  # Returns month name like "March"
        
    # Try to parse the string as a date
    try:
        # Assume format is like "March 2025" or similar
        date_parts = travel_dates_str.split()
        if len(date_parts) >= 1:
            # First part should be the month name
            month_name = date_parts[0].capitalize()
            # Check if it's a valid month name
            valid_months = [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ]
            if month_name in valid_months:
                return month_name
    except Exception as e:
        logger.error(f"Error parsing travel dates: {e}")
        
    # Return current month as fallback
    logger.warning(f"Could not parse travel dates '{travel_dates_str}', using current month")
    return datetime.now().strftime("%B")

def apply_time_decay(weight, interaction_time):
    """
    Apply time-based decay to an interaction weight.
    
    Args:
        weight: Base weight for the interaction
        interaction_time: Timestamp of the interaction
        
    Returns:
        Adjusted weight after time decay
    """
    # Get current time consistently as timezone-naive
    current_date = datetime.now().replace(tzinfo=None).date()
    
    # Handle string timestamps
    if isinstance(interaction_time, str):
        try:
            # Try to parse string timestamp and remove timezone
            timestamp = interaction_time.split("T")[0]  # Take just the date part
            interaction_date = datetime.strptime(timestamp, "%Y-%m-%d").date()
        except Exception as e:
            logger.error(f"Error parsing interaction timestamp: {e}")
            return weight  # Return original weight on error
    # Handle datetime objects
    elif hasattr(interaction_time, 'date'):
        try:
            interaction_date = interaction_time.replace(tzinfo=None).date()
        except Exception as e:
            logger.error(f"Error converting interaction time to date: {e}")
            return weight  # Return original weight on error
    else:
        # Fallback to current date
        logger.warning(f"Invalid interaction_time format: {type(interaction_time)}")
        return weight
        
    # Calculate days between dates
    days_ago = max(0, (current_date - interaction_date).days)
    decay = math.exp(-days_ago / 30)  # Exponential decay over 30 days
    
    return weight * decay

def get_numeric_value(obj, key, default=0):
    """
    Safely extract numeric values from MongoDB documents.
    
    Args:
        obj: MongoDB document or dictionary
        key: Key to extract
        default: Default value if key not found or value not numeric
        
    Returns:
        Numeric value (float or int)
    """
    if not obj or not isinstance(obj, dict):
        return default
        
    value = obj.get(key, default)
    
    # Handle MongoDB numeric types
    if isinstance(value, dict):
        if "$numberDouble" in value:
            return float(value["$numberDouble"])
        elif "$numberInt" in value:
            return int(value["$numberInt"])
        elif "$numberLong" in value:
            return int(value["$numberLong"])
            
    # Try to convert to float if it's a string
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
            
    # Return the value directly if it's already a number
    if isinstance(value, (int, float)):
        return value
        
    return default

def get_user_cached_recommendations(user_id):
    """
    Get all cached recommendation entries for a user, sorted by sequence.
    
    Args:
        user_id: User ID to get cache for
        
    Returns:
        List of cached recommendation entries sorted by sequence
    """
    try:
        # Find all cache entries for this user, sorted by sequence
        return list(recommendations_cache_collection.find(
            {"user_id": user_id, "_id": {"$ne": f"cache_lock_{user_id}"}}
        ).sort("sequence", 1))
    except Exception as e:
        logger.error(f"Error retrieving cached recommendations: {e}")
        return []

def clear_user_cache(user_id):
    """
    Clear all cached recommendations for a user.
    
    Args:
        user_id: User ID to clear cache for
        
    Returns:
        Number of entries deleted
    """
    try:
        result = recommendations_cache_collection.delete_many({"user_id": user_id})
        return result.deleted_count
    except Exception as e:
        logger.error(f"Error clearing user cache: {e}")
        return 0

def store_cache_entry(user_id, recommendations, sequence):
    """
    Store recommendations in cache with the given sequence number.
    
    Args:
        user_id: User ID
        recommendations: Recommendations data
        sequence: Sequence number for this cache entry
        
    Returns:
        True if successful, False otherwise
    """
    try:
        recommendations_cache_collection.insert_one({
            "user_id": user_id,
            "sequence": sequence,
            "recommendations": recommendations,
            "timestamp": datetime.now()
        })
        return True
    except Exception as e:
        logger.error(f"Error storing cache entry: {e}")
        return False
# --- PART 3: RECOMMENDATION ALGORITHM FUNCTIONS ---

def get_candidate_places(user_preferences, user_id, size=30):
    """
    Get candidate places for recommendations based on user preferences.
    Enhanced with improved semantic search for matching places to user preferences.
    
    Args:
        user_preferences: Dictionary containing user information with preferences or user_id
        user_id: User ID for fetching search history and interactions
        size: Maximum number of candidate places to return
        
    Returns:
        List of candidate places
    """
    logger.info(f"Finding candidate places for user {user_id}")
    
    # Helper function to safely get numeric values from MongoDB
    def get_safe_score(score):
        if isinstance(score, dict):
            # Handle MongoDB numeric types
            if "$numberDouble" in score:
                return float(score["$numberDouble"])
            if "$numberInt" in score:
                return float(int(score["$numberInt"]))
            if "$numberLong" in score:
                return float(int(score["$numberLong"]))
            return 0.0  # Default if it's an unrecognized dict
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0
    
    # FIXED: Make sure we have the full user document with preferences
    # If user_preferences doesn't have preferences, try to get the full user document
    if not isinstance(user_preferences, dict) or "preferences" not in user_preferences:
        # Try to fetch the user document from the database
        try:
            # Assuming users_collection is available in this scope
            user_doc = users_collection.find_one({"_id": user_id})
            if user_doc and "preferences" in user_doc:
                user_preferences = user_doc
                logger.info(f"Fetched user document from database for {user_id}")
            else:
                logger.warning(f"Could not find user preferences in database for {user_id}")
        except Exception as e:
            logger.error(f"Error fetching user from database: {e}")
    
    # Extract preferences from the user document
    if isinstance(user_preferences, dict) and "preferences" in user_preferences:
        preferences_obj = user_preferences.get("preferences", {})
        preferred_categories = preferences_obj.get("categories", [])
        preferred_tags = preferences_obj.get("tags", [])
    else:
        preferred_categories = []
        preferred_tags = []
    
    # Log preferences with the correct structure
    logger.info(f"User preferences - Categories: {preferred_categories}, Tags: {preferred_tags}")
    
    # If no preferences, return popular places
    if not preferred_categories and not preferred_tags:
        logger.warning(f"No user preferences found for user {user_id}, returning popular places")
        return list(places_collection.find().sort([("average_rating", -1)]).limit(size))
    
    # --- PART 1: VERY AGGRESSIVE FUZZY MATCHING ---
    all_places = list(places_collection.find())
    scored_places = []
    
    # Convert preferences to lowercase for comparison
    preferred_categories_lower = [cat.lower() for cat in preferred_categories if cat]
    preferred_tags_lower = [tag.lower() for tag in preferred_tags if tag]
    
    # Extract words from categories and tags for partial matching
    category_words = set()
    for cat in preferred_categories_lower:
        category_words.update(cat.split())
    
    tag_words = set()
    for tag in preferred_tags_lower:
        tag_words.update(tag.split())
    
    for place in all_places:
        score = 0
        place_name = place.get("name", "")
        place_category = place.get("category", "").lower() if place.get("category") else ""
        place_tags = [tag.lower() for tag in place.get("tags", [])] if isinstance(place.get("tags"), list) else []
        place_description = place.get("description", "").lower()
        
        # 1. Direct category match (highest weight)
        if place_category and any(cat == place_category for cat in preferred_categories_lower):
            score += 1.0
        
        # 2. Partial category match
        elif place_category:
            # Category contains or is contained by any preferred category
            if any(cat in place_category or place_category in cat for cat in preferred_categories_lower):
                score += 0.7
            # Word-level matching
            elif any(word in place_category for word in category_words):
                score += 0.5
        
        # 3. Tag matching - use both exact and partial
        if place_tags:
            # Direct tag matches
            exact_matches = sum(1 for tag in place_tags if tag in preferred_tags_lower)
            if exact_matches > 0:
                score += 0.8 * min(1.0, exact_matches / len(preferred_tags_lower))
            
            # Partial tag matches - check if any place tag contains or is contained by a preferred tag
            partial_matches = sum(1 for tag in place_tags 
                                if any(pref in tag or tag in pref 
                                      for pref in preferred_tags_lower))
            if partial_matches > 0:
                score += 0.5 * min(1.0, partial_matches / len(preferred_tags_lower))
            
            # Word-level tag matching
            word_matches = sum(1 for tag in place_tags 
                             if any(word in tag for word in tag_words))
            if word_matches > 0:
                score += 0.3 * min(1.0, word_matches / len(tag_words))

        
        # 4. Check description for keywords (bonus match)
        if place_description:
            cat_matches = sum(1 for cat in preferred_categories_lower if cat in place_description)
            tag_matches = sum(1 for tag in preferred_tags_lower if tag in place_description)
            
            if cat_matches > 0 or tag_matches > 0:
                score += 0.2  # Small bonus for description matches
        
        # 5. EXTRA AGGRESSIVE: Give every place at least a minimal score to ensure some matches
        if score == 0:
            score = 0.01
        
        scored_places.append((place, score))
    
    # FIXED: Sort by score using safe extraction of score value
    scored_places.sort(key=lambda x: get_safe_score(x[1]), reverse=True)
    category_tag_places = [place for place, score in scored_places if get_safe_score(score) > 0.01]  # Filter minimal scores
    
    # Limit to ensure we don't have too many low-quality matches
    category_tag_places = category_tag_places[:min(len(category_tag_places), size)]
    
    logger.info(f"Found {len(category_tag_places)} places with direct category/tag matching for user {user_id}")
    
    # Continue with the rest of the function (semantic search)...
    # --- PART 2: SEMANTIC SEARCH BASED ON RECENT QUERIES (40%) ---
    semantic_places = []
    
    # Check if NLP is available with more robust verification
    nlp_available = False
    fallback_level = 0
    
    try:
        # Level 1: Check if nlp exists and has vectors
        if nlp and not isinstance(nlp, DummyNLP):
            test_doc = nlp("test")
            if hasattr(test_doc, 'vector_norm') and test_doc.vector_norm > 0:
                nlp_available = True
                logger.info("Using full spaCy NLP with word vectors")
            else:
                fallback_level = 1
                logger.warning("spaCy available but word vectors missing, using fallback level 1")
        else:
            fallback_level = 2
            logger.warning("spaCy not available, using fallback level 2")
    except Exception as e:
        fallback_level = 3
        logger.error(f"Error checking NLP availability: {e}, using fallback level 3")
    
    try:
        # Fetch recent search queries for this user
        search_queries = list(search_queries_collection.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(5))
        
        # Extract keywords from search queries with appropriate fallbacks
        search_keywords = set()

        for query_doc in search_queries:
            # Use existing keywords if available
            if "keywords" in query_doc and query_doc["keywords"]:
                for keyword in query_doc["keywords"]:
                    if keyword and len(keyword) > 2:
                        search_keywords.add(keyword.lower())
            else:
                query_text = query_doc.get("query", "")
                if not query_text:
                    continue
                
                # Different keyword extraction based on fallback level
                if fallback_level == 0:
                    # Full NLP with POS tagging
                    doc = nlp(query_text.lower())
                    for token in doc:
                        if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop and len(token.text) > 2:
                            search_keywords.add(token.text)
                elif fallback_level == 1:
                    # spaCy available but no vectors - use basic POS if available, otherwise tokenize
                    try:
                        doc = nlp(query_text.lower())
                        for token in doc:
                            if not token.is_stop and len(token.text) > 2:
                                search_keywords.add(token.text)
                    except:
                        # Tokenize and filter stopwords
                        stopwords = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
                        for word in query_text.lower().split():
                            if word not in stopwords and len(word) > 2:
                                search_keywords.add(word)
                else:
                    # Basic tokenization for levels 2 and 3
                    stopwords = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
                    for word in query_text.lower().split():
                        if word not in stopwords and len(word) > 2:
                            search_keywords.add(word)
        
        search_keywords_list = list(search_keywords)  # Convert to list for consistent ordering
        logger.info(f"Extracted {len(search_keywords_list)} search keywords for user {user_id}")
        
        if search_keywords_list:
            # Calculate semantic similarity scores based on fallback level
            scored_semantic_places = []
            
            # Fallback level determines matching approach
            if fallback_level == 0:
                # Full semantic matching with spaCy word vectors
                for place in all_places:
                    place_id = place["_id"]

                    # Skip if place is already highly ranked in category/tag matches
                    if place in category_tag_places[:int(size * 0.3)]:
                        continue
                    
                    # Initialize scoring components
                    tag_similarity = 0.0
                    tag_match_count = 0
                    description_similarity = 0.0
                    description_match_count = 0
                    
                    # Get place data
                    tags = place.get("tags", [])
                    description = place.get("description", "")
                    
                    # Process tags with nlp
                    for keyword in search_keywords_list:
                        keyword_doc = nlp(keyword.lower())
                        
                        # Check each tag for similarity
                        for tag in tags:
                            try:
                                tag_doc = nlp(tag.lower())
                                # FIXED: Check if both docs have valid vectors before comparing
                                if (hasattr(keyword_doc, 'vector_norm') and keyword_doc.vector_norm > 0 and 
                                    hasattr(tag_doc, 'vector_norm') and tag_doc.vector_norm > 0):
                                    similarity = keyword_doc.similarity(tag_doc)
                                    
                                    # Count significant matches
                                    if similarity > 0.6:  # Threshold for semantic match
                                        tag_similarity += similarity
                                        tag_match_count += 1
                                else:
                                    # Fallback to exact/partial matching
                                    if keyword.lower() in tag.lower() or tag.lower() in keyword.lower():
                                        tag_similarity += 0.7
                                        tag_match_count += 1
                            except Exception as e:
                                continue  # Skip this tag if error
                        
                        # Process description with nlp
                        if description:
                            try:
                                # Process full description
                                desc_doc = nlp(description.lower())
                                
                                # FIXED: Check if both docs have valid vectors before comparing
                                if (hasattr(keyword_doc, 'vector_norm') and keyword_doc.vector_norm > 0 and 
                                    hasattr(desc_doc, 'vector_norm') and desc_doc.vector_norm > 0):
                                    similarity = keyword_doc.similarity(desc_doc)
                                    
                                    # Check exact keyword match in description (bonus)
                                    if keyword.lower() in description.lower():
                                        description_similarity += max(similarity, 0.7)  # At least 0.7 for exact match
                                        description_match_count += 1
                                    elif similarity > 0.5:  # Lower threshold for description
                                        description_similarity += similarity
                                        description_match_count += 1
                                else:
                                    # Fallback to exact matching
                                    if keyword.lower() in description.lower():
                                        description_similarity += 0.7
                                        description_match_count += 1
                            except Exception as e:
                                logger.debug(f"Error comparing description: {str(e)}")

                    # Calculate final semantic score
                    semantic_score = 0.0
                    
                    # Tag component (60% weight)
                    tag_component = 0.0
                    if tag_match_count > 0:
                        tag_component = (tag_similarity / tag_match_count) * 0.6
                    
                    # Description component (40% weight)
                    desc_component = 0.0
                    if description_match_count > 0:
                        desc_component = (description_similarity / description_match_count) * 0.4
                    
                    # Combined score
                    semantic_score = tag_component + desc_component
                    
                    # Only include if score is significant
                    if semantic_score > 0.3:
                        scored_semantic_places.append((place, semantic_score))
            else:
                # Text-based fallback matching (for fallback levels 1-3)
                for place in all_places:
                    # Skip if place is already highly ranked in category/tag matches
                    if place in category_tag_places[:int(size * 0.3)]:
                        continue
                    
                    match_score = 0.0
                    tags = place.get("tags", [])
                    description = place.get("description", "")
                    
                    # Tag matching (60% weight)
                    tag_matches = 0
                    for keyword in search_keywords_list:
                        for tag in tags:
                            # Exact or partial match
                            if keyword.lower() in tag.lower() or tag.lower() in keyword.lower():
                                tag_matches += 1
                                break
                    
                    if tag_matches > 0:
                        # Scale by ratio of matched keywords
                        tag_score = min(1.0, tag_matches / len(search_keywords_list))
                        match_score += tag_score * 0.6  # 60% weight
                    
                    # Description matching (40% weight)
                    if description:
                        desc_matches = 0
                        for keyword in search_keywords_list:
                            if keyword.lower() in description.lower():
                                desc_matches += 1
                        
                        if desc_matches > 0:
                            # Scale by ratio of matched keywords
                            desc_score = min(1.0, desc_matches / len(search_keywords_list))
                            match_score += desc_score * 0.4  # 40% weight
                    
                    if match_score > 0.2:  # Lower threshold for text matching
                        scored_semantic_places.append((place, match_score))
            
            # FIXED: Sort by score using safe extraction of score value
            scored_semantic_places.sort(key=lambda x: get_safe_score(x[1]), reverse=True)
            semantic_places = [place for place, _ in scored_semantic_places]
            logger.info(f"Found {len(semantic_places)} places via search keywords (fallback level: {fallback_level})")
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        # If semantic search fails, we'll just use the category/tag results
    
    # --- PART 3: COMBINE RESULTS ---
    # Calculate counts for each source
    category_tag_count = min(len(category_tag_places), int(size * 0.6))
    semantic_count = min(len(semantic_places), size - category_tag_count)
    
    # Combine places with no duplicates
    candidate_places = []
    added_ids = set()
    
    # Add category/tag places first (60%)
    for place in category_tag_places[:category_tag_count]:
        place_id = place["_id"]
        if place_id not in added_ids:
            candidate_places.append(place)
            added_ids.add(place_id)
    
    # Add semantic places (40%)
    for place in semantic_places[:semantic_count]:
        place_id = place["_id"]
        if place_id not in added_ids and len(candidate_places) < size:
            candidate_places.append(place)
            added_ids.add(place_id)
    
    # If we don't have enough candidates, add some top-rated places
    if len(candidate_places) < size:
        additional_places = list(
            places_collection.find({"_id": {"$nin": list(added_ids)}})
            .sort([("average_rating", -1)])
            .limit(size - len(candidate_places))
        )
        candidate_places.extend(additional_places)
        logger.info(f"Added {len(additional_places)} additional places based on popularity")
    
    logger.info(f"Returning {len(candidate_places)} total candidate places for user {user_id}")
    return candidate_places

import math
from datetime import datetime, timedelta

def get_collaborative_recommendations(user_id, target_count=39, excluded_place_ids=None):
    """
    Get place recommendations based on similar users' interactions,
    with caching of similar users for improved performance.
    
    Args:
        user_id: User ID to get recommendations for
        target_count: Number of recommendations to generate
        excluded_place_ids: Place IDs to exclude
        
    Returns:
        List of place IDs recommended through collaborative filtering
    """
    # Helper function to safely extract numeric values
    def ensure_float(value):
        if isinstance(value, dict):
            # Handle MongoDB numeric types
            if "$numberDouble" in value:
                return float(value["$numberDouble"])
            if "$numberInt" in value:
                return float(int(value["$numberInt"]))
            if "$numberLong" in value:
                return float(int(value["$numberLong"]))
            return 0.0  # Default if it's an unrecognized dict
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
            
    try:
        logger.info(f"Finding collaborative recommendations for user {user_id}")
        
        if excluded_place_ids is None:
            excluded_place_ids = []
            
        user = users_collection.find_one({"_id": user_id})
        if not user:
            logger.warning(f"User {user_id} not found")
            return []

        # Find similar users with caching optimization
        similar_users = find_similar_users(user_id)
        logger.info(f"Found {len(similar_users)} similar users for user {user_id}")
        
        if not similar_users:
            return []
        
        # Get similar user IDs and create similarity map
        similar_user_ids = [u["user_id"] for u in similar_users]
        similarity_map = {u["user_id"]: ensure_float(u["similarity"]) for u in similar_users}  # Use ensure_float here

        # Define weights for different interaction types
        action_weights = {
            "like": 5,
            "save": 4,
            "share": 3,
            "view": 2,
        }

        # Get current time consistently - ensure it's timezone-naive
        current_time = datetime.now().replace(tzinfo=None)
        
        # Query interactions efficiently with a single query
        similar_interactions = list(interactions_collection.find({
            "user_id": {"$in": similar_user_ids},
            "place_id": {"$nin": excluded_place_ids}
        }))

        # Track recommended places with scores
        place_scores = {}

        # Get existing interactions for user
        user_interactions = {}
        for i in interactions_collection.find({"user_id": user_id}):
            if "place_id" in i and "interaction_type" in i:
                user_interactions[i["place_id"]] = i["interaction_type"]

        # Process interactions from similar users
        for interaction in similar_interactions:
            # Skip if place_id or interaction_type is missing
            if "place_id" not in interaction or "interaction_type" not in interaction:
                continue

            place_id = interaction["place_id"]
            similar_user_id = interaction["user_id"]
            action = interaction["interaction_type"]
            
            # Get user similarity score - ensure it's a standard float
            user_similarity = ensure_float(similarity_map.get(similar_user_id, 0.5))  # Default 0.5 if missing
            
            # Skip low similarity users
            if user_similarity < 0.3:
                continue

            # Skip if user already dislikes this place
            if place_id in user_interactions and user_interactions[place_id] == "dislike":
                continue

            # Calculate time decay - handle datetime timezone issues
            try:
                interaction_time = interaction.get("timestamp", current_time)
                
                # Convert string timestamps to datetime
                if isinstance(interaction_time, str):
                    try:
                        # Parse to datetime and explicitly remove timezone info
                        interaction_time = datetime.fromisoformat(interaction_time.replace('Z', '')).replace(tzinfo=None)
                    except:
                        # If parsing fails, use current time
                        interaction_time = current_time
                
                # Ensure any datetime is timezone-naive
                if hasattr(interaction_time, 'tzinfo') and interaction_time.tzinfo is not None:
                    interaction_time = interaction_time.replace(tzinfo=None)
                
                # Calculate days difference
                time_diff = current_time - interaction_time
                days_ago = max(1, time_diff.days)
                time_decay = 1.0 / (1 + math.log(days_ago))
                
            except Exception as time_error:
                # If any time calculation fails, use default decay
                logger.warning(f"Time decay calculation error: {time_error}, using default")
                time_decay = 0.5
            
            # Get action weight
            weight = action_weights.get(action, 1)  # Default weight of 1 for unknown actions
            
            # Calculate final score
            final_score = weight * time_decay * user_similarity

            # Only add positively scored places
            if final_score > 0:
                if place_id not in place_scores:
                    place_scores[place_id] = 0
                place_scores[place_id] += final_score

        # Sort places by score - using the helper function for comparison
        sorted_places = sorted(place_scores.items(), key=lambda x: ensure_float(x[1]), reverse=True)
        top_place_ids = [place_id for place_id, _ in sorted_places[:target_count*2]]
        
        if not top_place_ids:
            return []
        
        # Log the number of recommendations found
        logger.info(f"Found {len(top_place_ids[:target_count])} collaborative recommendations for user {user_id}")
        
        # Return just the place IDs to match the expected format in generate_final_recommendations
        return top_place_ids[:target_count]
        
    except Exception as e:
        logger.error(f"Error in collaborative filtering: {str(e)}")
        return []
def calculate_similarity_score(user1, user2):
    try:
        # Get user IDs for better logging
        user1_id = user1.get("_id", "unknown")
        user2_id = user2.get("_id", "unknown")
        
        # Get preferences
        prefs1 = user1.get("preferences", {})
        prefs2 = user2.get("preferences", {})
        
        if not prefs1 or not prefs2:
            return 0.3  # Default modest similarity when preferences missing
        
        # 1. Category similarity (weighted 40%)
        cats1 = set(prefs1.get("categories", []))
        cats2 = set(prefs2.get("categories", []))
        
        category_jaccard = 0
        if cats1 and cats2:
            # Jaccard similarity for categories
            category_jaccard = len(cats1.intersection(cats2)) / max(len(cats1.union(cats2)), 1)
            
        # 2. Tag similarity (weighted 40%)
        tags1 = set(prefs1.get("tags", []))
        tags2 = set(prefs2.get("tags", []))
        
        tag_jaccard = 0
        if tags1 and tags2:
            # Jaccard similarity for tags
            tag_jaccard = len(tags1.intersection(tags2)) / max(len(tags1.union(tags2)), 1)
        
        # 3. Activity level similarity (weighted 20%) 
        activity1 = prefs1.get("activity_level", "medium")
        activity2 = prefs2.get("activity_level", "medium")
        
        activity_score = 0
        if activity1 == activity2:
            activity_score = 1.0
        elif (activity1 in ["high", "medium"] and activity2 in ["high", "medium"]) or \
             (activity1 in ["medium", "low"] and activity2 in ["medium", "low"]):
            # Adjacent activity levels
            activity_score = 0.5
        
        # Calculate total score with weightings
        category_contribution = category_jaccard * 0.4
        tag_contribution = tag_jaccard * 0.4
        activity_contribution = activity_score * 0.2
        
        # Compute final score - important changes here
        # We weight the scores differently to avoid the need for normalization
        # and ensure we get higher similarity values
        similarity_score = category_contribution + tag_contribution + activity_contribution
        
        # Boost the score to increase matches - increased boost factor from 1.2 to 1.5
        # to ensure more users cross the threshold
        boosted_score = min(1.0, similarity_score * 1.5)
        
        # More detailed logging with the actual components
        logger.debug(f"Similarity details for {user1_id}-{user2_id}: " +
                   f"cats: {category_jaccard:.2f}*0.4={category_contribution:.2f}, " +
                   f"tags: {tag_jaccard:.2f}*0.4={tag_contribution:.2f}, " +
                   f"activity: {activity_score:.2f}*0.2={activity_contribution:.2f}, " +
                   f"total: {similarity_score:.2f}, boosted: {boosted_score:.2f}")
        
        return boosted_score
        
    except Exception as e:
        logger.error(f"Error calculating user similarity: {str(e)}")
        return 0.3  # Default modest similarity on error
def find_similar_users(user_id, min_similarity=0.15, max_users=40):  # Lowered threshold from 0.25 to 0.15
    """
    Find users similar to the given user based on preferences and interactions,
    with caching for improved performance.
    
    Args:
        user_id: ID of the user to find similar users for
        min_similarity: Minimum similarity score to include a user
        max_users: Maximum number of similar users to return
        
    Returns:
        List of similar user IDs with similarity scores
    """
    # Helper function to safely extract numeric values
    def get_safe_score(item):
        score = item["similarity"]
        if isinstance(score, dict):
            # Handle MongoDB numeric types
            if "$numberDouble" in score:
                return float(score["$numberDouble"])
            if "$numberInt" in score:
                return float(int(score["$numberInt"]))
            if "$numberLong" in score:
                return float(int(score["$numberLong"]))
            return 0.0  # Default if it's an unrecognized dict
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0
    
    try:
        # Check if we have cached similar users
        cached_similar = similar_users_cache.find_one({"user_id": user_id})
        
        if cached_similar and "similar_users" in cached_similar:
            logger.info(f"Using cached similar users for {user_id}")
            return cached_similar["similar_users"]
        
        # Get user document
        user_doc = users_collection.find_one({"_id": user_id})
        if not user_doc:
            logger.warning(f"User {user_id} not found when finding similar users")
            return []
            
        # Get user preferences from the correct location
        user_prefs = user_doc.get("preferences", {})
        user_categories = set(user_prefs.get("categories", []))
        user_tags = set(user_prefs.get("tags", []))
        
        # Get all other users
        all_users = list(users_collection.find({"_id": {"$ne": user_id}}))
        
        similar_users = []
        all_scored_users = []  # Track all users with their scores for fallback
        
        for other_user in all_users:
            other_id = other_user["_id"]
            
            # Calculate base similarity score between users
            similarity_score = calculate_similarity_score(user_doc, other_user)
            
            # Track all users with their scores for potential fallback
            all_scored_users.append({
                "user_id": other_id,
                "similarity": similarity_score
            })
            
            # Only include users above minimum similarity in the main list
            # Safe comparison using converted value
            safe_score = get_safe_score({"similarity": similarity_score})
            if safe_score >= min_similarity:
                similar_users.append({
                    "user_id": other_id,
                    "similarity": similarity_score
                })
                logger.info(f"Found similar user {other_id} with score {safe_score:.2f}")
        
        # Sort by similarity (descending) and take top users
        # FIXED: Use safe score extraction for sorting
        similar_users.sort(key=get_safe_score, reverse=True)
        similar_users = similar_users[:max_users]
        
        # If no similar users above threshold, use fallback
        if len(similar_users) == 0 and len(all_scored_users) > 0:
            logger.info(f"No users above similarity threshold {min_similarity}. Using best available matches.")
            
            # Sort and take top 5 regardless of threshold
            # FIXED: Use safe score extraction for sorting
            all_scored_users.sort(key=get_safe_score, reverse=True)
            similar_users = all_scored_users[:5]
            
            logger.info(f"Using {len(similar_users)} fallback similar users with lower similarity")
            
            # Log the top fallback user
            if similar_users:
                top_user = similar_users[0]
                top_safe_score = get_safe_score(top_user)
                logger.info(f"Top fallback user: {top_user['user_id']} with similarity {top_safe_score:.2f}")
        
        # Cache the result
        similar_users_cache.update_one(
            {"user_id": user_id},
            {"$set": {
                "user_id": user_id,
                "similar_users": similar_users,
                "timestamp": datetime.now()
            }},
            upsert=True
        )
        
        # Log summary
        logger.info(f"Found {len(similar_users)} similar users for user {user_id}")
        
        return similar_users
        
    except Exception as e:
        logger.error(f"Error finding similar users for {user_id}: {e}")
        return []
def get_discovery_places(user_id, limit=10):
    """Get places outside the user's normal patterns for discovery"""
    try:
        user_prefs = get_user_travel_preferences(user_id)
        if not user_prefs:
            return []

        preferred_categories = user_prefs.get("preferred_categories", [])
        preferred_tags = user_prefs.get("preferred_tags", [])

        # Define ensure_float helper function
        def ensure_float(value):
            if isinstance(value, dict):
                # Handle MongoDB numeric types
                if "$numberDouble" in value:
                    return float(value["$numberDouble"])
                if "$numberInt" in value:
                    return float(int(value["$numberInt"]))
                if "$numberLong" in value:
                    return float(int(value["$numberLong"]))
                return 0.0  # Default if it's an unrecognized dict
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        # Find places in different categories but highly rated
        # Use MongoDB's aggregate to properly compare ratings
        discovery_pipeline = [
            {
                "$match": {
                    "category": {"$nin": preferred_categories}
                }
            },
            # Use $convert to safely convert rating to double
            {
                "$addFields": {
                    "numeric_rating": {
                        "$convert": {
                            "input": "$average_rating",
                            "to": "double",
                            "onError": 0,
                            "onNull": 0
                        }
                    }
                }
            },
            {
                "$match": {
                    "numeric_rating": {"$gte": 4.0}
                }
            },
            {
                "$sort": {"numeric_rating": -1}
            },
            {
                "$limit": limit * 2
            }
        ]

        discovery_places = list(places_collection.aggregate(discovery_pipeline))

        # If we don't have enough, try a broader search
        if len(discovery_places) < limit:
            fallback_pipeline = [
                {
                    "$match": {
                        "category": {"$nin": preferred_categories}
                    }
                },
                # Use $convert for the fallback pipeline too
                {
                    "$addFields": {
                        "numeric_rating": {
                            "$convert": {
                                "input": "$average_rating",
                                "to": "double",
                                "onError": 0,
                                "onNull": 0
                            }
                        }
                    }
                },
                {
                    "$sort": {"numeric_rating": -1}
                },
                {
                    "$limit": limit * 2
                }
            ]

            fallback_places = list(places_collection.aggregate(fallback_pipeline))

            # Add any new places not already in discovery_places
            existing_ids = set(p["_id"] for p in discovery_places)
            for place in fallback_places:
                if place["_id"] not in existing_ids:
                    discovery_places.append(place)
                    if len(discovery_places) >= limit * 2:
                        break

        # Randomize the order for more variety in recommendations
        if discovery_places:
            random.shuffle(discovery_places)

        logger.info(f"Found {len(discovery_places[:limit])} discovery places for user {user_id}")
        return discovery_places[:limit]
    except Exception as e:
        logger.error(f"Error getting discovery places: {str(e)}")
        return []
def calculate_personalization_score(place, user_id, user_prefs):
    """
    Calculate personalization score for a place based on user preferences.
    
    Scoring components:
    - Category match (35%): Direct match (1.0) or partial/substring match (0.7)
    - Tag match (25%): Proportional to number of matching tags
    - Rating factor (15%): Normalized place average_rating (0-5 scale)
    - User interaction history (10%): Based on interaction_type (like, save, share, view)
    - Review factors (15%): Based on review sentiment, likes/dislikes, and count
    
    The final score is weighted: (category*0.35 + tags*0.25 + rating*0.15 + interactions*0.1 + reviews*0.15)
    
    Args:
        place: Place document
        user_id: User ID
        user_prefs: User preferences dictionary
        
    Returns:
        Personalization score between 0 and 1
    """
    import re
    import math
    
    # Helper function to safely extract numeric values from MongoDB data
    def extract_numeric(value, default=0):
        try:
            if isinstance(value, dict):
                # Handle MongoDB numeric types
                if "$numberDouble" in value:
                    val = float(value["$numberDouble"])
                    # Check for invalid JSON values
                    if math.isnan(val) or math.isinf(val):
                        return default
                    return val
                if "$numberInt" in value:
                    return float(int(value["$numberInt"]))
                if "$numberLong" in value:
                    return float(int(value["$numberLong"]))
                # Skip MongoDB date objects
                if "$date" in value:
                    return default
                return default
            
            # Handle direct numeric types
            if isinstance(value, (int, float)):
                # Check for invalid JSON values
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    return default
                return float(value)
            
            # Try converting string to float
            val = float(value)
            # Check for invalid JSON values
            if math.isnan(val) or math.isinf(val):
                return default
            return val
        except (TypeError, ValueError):
            return default

    try:
        # 1. Category matching (35% of score) - reduced from 40%
        category_score = 0
        place_category = place.get("category", "").lower()
        preferred_categories = [cat.lower() for cat in user_prefs.get("preferred_categories", [])]
        
        if preferred_categories:
            # Direct category match
            if place_category in preferred_categories:
                category_score = 1.0
            else:
                # Check for partial matches (e.g., "beach resort" contains "beach")
                for category in preferred_categories:
                    if category in place_category or place_category in category:
                        category_score = 0.7
                        break
        else:
            # No preferred categories, neutral score
            category_score = 0.5
            
        # 2. Tag matching (25% of score) - reduced from 30%
        tag_score = 0
        place_tags = [tag.lower() for tag in place.get("tags", [])]
        preferred_tags = [tag.lower() for tag in user_prefs.get("preferred_tags", [])]
        
        if preferred_tags and place_tags:
            matching_tags = set(place_tags).intersection(set(preferred_tags))
            tag_score = len(matching_tags) / max(len(preferred_tags), 1)
        else:
            # No tags to compare, neutral score
            tag_score = 0.5
            
        # 3. Rating factor (15% of score) - reduced from 20%
        # FIXED: Changed from rating to average_rating
        raw_rating = place.get("average_rating", 0)
        rating_value = extract_numeric(raw_rating, 0)  # Extract numeric value safely
        rating_score = min(rating_value / 5.0, 1.0)  # Normalize to 0-1
            
        # 4. User interaction history (10% of score)
        interaction_score = 0.5  # Default neutral score
        
        # Look up past interactions from interactions collection
        user_interactions = list(interactions_collection.find({
            "user_id": user_id,
            "place_id": place.get("_id", "")
        }))
        
        # Process all interactions for this user and place
        if user_interactions:
            # Track interaction types and view count
            interaction_types = set()
            view_count = 0
            
            for interaction in user_interactions:
                interaction_type = interaction.get("interaction_type", "")
                if interaction_type == "like":
                    interaction_types.add("like")
                elif interaction_type == "save":
                    interaction_types.add("save")
                elif interaction_type == "share":
                    interaction_types.add("share") 
                elif interaction_type == "view":
                    view_count += 1
            
            # Assign score based on interaction types (prioritized)
            if "like" in interaction_types:
                interaction_score = 0.9  # Highest score for likes
            elif "save" in interaction_types:
                interaction_score = 0.8  # High score for saves
            elif "share" in interaction_types:
                interaction_score = 0.7  # Good score for shares
            elif view_count > 3:
                interaction_score = 0.6  # Moderate score for multiple views
        
        # 5. Review factors (15% of score) - NEW component
        review_score = 0.5  # Default neutral score
        
        # Look up reviews for this place
        place_reviews = list(reviews_collection.find({"place_id": place.get("_id", "")}))
        
        if place_reviews:
            # 5.1 Social proof from likes and dislikes (enhanced)
            total_likes = 0
            total_dislikes = 0
            for review in place_reviews:
                # Extract likes and dislikes using existing extract_numeric helper
                likes_count = extract_numeric(review.get("likes", 0))
                dislikes_count = extract_numeric(review.get("dislikes", 0))
                total_likes += likes_count
                total_dislikes += dislikes_count
            
            # Calculate social proof ratio (with protection against division by zero)
            total_interactions = total_likes + total_dislikes
            if total_interactions > 0:
                likes_ratio = total_likes / total_interactions
                # Scale the ratio: 0% positive = 0.1, 50% positive = 0.5, 100% positive = 0.9
                likes_factor = 0.1 + (likes_ratio * 0.8)
            else:
                # No likes or dislikes, use neutral score
                likes_factor = 0.5
            
            # Add volume factor to give more weight to places with more engagement
            # More engagement (whether positive or negative) = more reliable signal
            volume_boost = min(math.log10(1 + total_interactions) / 10.0, 0.1)
            social_proof_factor = likes_factor + volume_boost
            
            # 5.2 Pattern-based sentiment analysis
            # Define positive and negative word lists - EXPANDED
            positive_words = [
                # Original words
                "amazing", "excellent", "great", "good", "wonderful", "awesome", 
                "beautiful", "enjoyed", "recommend", "fantastic", "love", "best",
                "perfect", "impressive", "breathtaking", "stunning",
                
                # Additional travel-specific positive words
                "authentic", "scenic", "convenient", "spacious", "delicious",
                "friendly", "helpful", "clean", "comfortable", "peaceful",
                "relaxing", "paradise", "unforgettable", "magical", "charming",
                "luxurious", "tranquil", "picturesque", "idyllic", "spectacular",
                "worth", "gem", "hidden", "treasure", "affordable", "value",
                "welcoming", "cozy", "unique", "convenient", "safe", "accessible",
                "fascinating", "interesting", "informative", "entertaining",
                "immersive", "refreshing", "rejuvenating", "serene", "pleasant",
                "memorable", "enchanting", "lovely", "delightful", "superb",
                "exceptional", "outstanding", "terrific", "marvelous", "fabulous",
                "paradise", "heaven", "ideal", "perfect", "dream", "romantic",
                "exotic", "sophisticated", "elegant", "stylish", "modern",
                "authentic", "genuine", "traditional", "cultural", "historical",
                "satisfied", "satisfying", "enjoyable", "fun", "exciting",
                "therapeutic", "professional", "courteous", "attentive", "organized",
                "efficient", "quick", "prompt", "convenient", "central",
                "incredible", "generous", "reasonable", "affordable", "bargain",
                "gorgeous", "majestic", "unbelievable", "astonishing", "awe-inspiring"
            ]
            
            negative_words = [
                # Original words
                "disappointing", "terrible", "bad", "worst", "avoid", "horrible", 
                "poor", "waste", "unfortunate", "awful", "mediocre", "not worth",
                "overpriced", "dirty", "crowded", "rude",
                
                # Additional travel-specific negative words
                "overrated", "uncomfortable", "boring", "unsafe", "dangerous",
                "expensive", "noisy", "unreliable", "inconvenient", "cramped",
                "smelly", "sketchy", "rundown", "outdated", "neglected",
                "unpleasant", "disappointing", "lackluster", "touristy", "scam",
                "ripoff", "tourist-trap", "underwhelming", "unhelpful", "unfriendly",
                "slow", "inefficient", "frustrating", "disorganized", "chaotic",
                "bland", "tasteless", "greasy", "cold", "stale", "unsanitary",
                "unhygienic", "questionable", "suspicious", "misleading", "dishonest",
                "unprofessional", "amateurish", "sloppy", "careless", "neglectful",
                "indifferent", "apathetic", "uninterested", "unresponsive", "remote",
                "isolated", "inaccessible", "difficult", "complicated", "confusing",
                "unimpressive", "ordinary", "forgettable", "unremarkable", "generic",
                "tacky", "cheesy", "kitschy", "gimmicky", "artificial", "fake",
                "appalling", "dreadful", "filthy", "disgusting", "unacceptable",
                "pathetic", "ridiculous", "absurd", "terrible", "atrocious",
                "miserable", "regret", "regrettable", "annoying", "irritating",
                "infuriating", "outrageous", "disgraceful", "shameful", "abysmal",
                "disastrous", "nightmarish", "horrific", "creepy", "scary",
                "distressing", "unsettling", "worrying", "concerning", "problematic"
            ]
            
            # Create patterns with word boundaries
            positive_patterns = [r'\b' + word + r'\b' for word in positive_words]
            negative_patterns = [r'\b' + word + r'\b' for word in negative_words]
            
            # Add negative context patterns - EXPANDED
            negative_contexts = [
                # Original patterns
                r'not\s+\w*\s*(?:good|great|nice)', 
                r'too\s+(?:expensive|crowded|busy)',
                r'(?:never|wouldn\'t)\s+(?:recommend|return)',
                
                # Additional patterns
                r'far\s+from\s+(?:everything|anything)',
                r'nothing\s+(?:special|interesting|unique)',
                r'lack\s+of\s+(?:cleanliness|service|amenities|facilities)',
                r'didn\'t\s+(?:enjoy|like|appreciate)',
                r'waste\s+of\s+(?:money|time)',
                r'could\s+(?:be|have been)\s+better',
                r'not\s+(?:worth|clean|friendly|helpful)',
                r'(?:barely|hardly)\s+(?:edible|acceptable|adequate)',
                r'(?:completely|totally)\s+(?:disappointed|let down)',
                r'bad\s+(?:experience|service|food|accommodation)',
                r'hard\s+to\s+(?:find|access|reach)',
                r'no\s+(?:charm|character|atmosphere)',
                r'better\s+(?:places|options|alternatives)\s+(?:nearby|elsewhere|available)',
                r'avoid\s+(?:this|at all costs)',
                r'expected\s+(?:more|better)',
                r'not\s+as\s+(?:good|nice|clean|friendly)\s+as',
                r'wouldn\'t\s+(?:stay|visit|eat)\s+(?:here|there|again)',
                r'(?:should|needs to)\s+(?:improve|change|update)',
                r'(?:way|far)\s+(?:too|below)\s+(?:expensive|pricey|overpriced)',
                r'(?:false|misleading)\s+(?:advertising|description|photos)'
            ]
            
            sentiment_sum = 0
            reviews_with_text = 0
            
            for review in place_reviews:
                if review.get("review_text"):
                    reviews_with_text += 1
                    text = review.get("review_text", "").lower()
                    
                    # Count positive patterns
                    pos_count = 0
                    for pattern in positive_patterns:
                        matches = re.findall(pattern, text)
                        pos_count += len(matches)
                    
                    # Count negative patterns
                    neg_count = 0
                    for pattern in negative_patterns:
                        matches = re.findall(pattern, text)
                        neg_count += len(matches)
                    
                    # Add negative context matches
                    for pattern in negative_contexts:
                        matches = re.findall(pattern, text)
                        neg_count += len(matches)
                    
                    # Add weighting for review length
                    length_factor = min(len(text.split()) / 50, 1.0)  # Cap at reviews of 50+ words
                    
                    # Calculate final sentiment for this review
                    if pos_count + neg_count > 0:
                        # Calculate base sentiment (-1 to +1 scale)
                        base_sentiment = (pos_count - neg_count) / (pos_count + neg_count)
                        # Weight longer reviews more heavily
                        weighted_sentiment = base_sentiment * (0.7 + (length_factor * 0.3))
                        # Normalize to 0-1 scale
                        normalized_sentiment = (weighted_sentiment + 1) / 2
                        sentiment_sum += normalized_sentiment
                    else:
                        # No sentiment detected, use neutral score slightly weighted by length
                        sentiment_sum += 0.5 * (1 + (length_factor * 0.1))
            
            # Calculate average sentiment
            sentiment_score = sentiment_sum / reviews_with_text if reviews_with_text > 0 else 0.5
            
            # 5.3 Review count as a reliability measure
            # 0 reviews = 0.5, 5 reviews = 0.7, 10+ reviews = 0.9
            count_factor = 0.5 + min(len(place_reviews) / 20.0, 0.4)
            
            # Combine all review factors (updated weights)
            # Give more weight to social proof factor (incorporating likes/dislikes)
            review_score = (social_proof_factor * 0.45) + (sentiment_score * 0.35) + (count_factor * 0.2)
        
        # Calculate final weighted score with review component
        final_score = (
            (category_score * 0.35) +   # Reduced from 0.4
            (tag_score * 0.25) +        # Reduced from 0.3
            (rating_score * 0.15) +     # Reduced from 0.2
            (interaction_score * 0.1) + # Unchanged
            (review_score * 0.15)       # New component
        )
        
        return final_score
        
    except Exception as e:
        logger.error(f"Error calculating personalization score: {e}")
        return 0.5  # Return neutral score on error
def rank_places(candidate_places, user_id):
    """
    Rank places based on user engagement and popularity metrics.
    
    Args:
        candidate_places: List of place documents to rank
        user_id: User ID for personalization
        
    Returns:
        List of places sorted by relevance score
    """
    # Helper function to extract numeric value from MongoDB document fields
    def extract_number(value):
        if isinstance(value, dict):
            # Handle MongoDB numeric types
            if "$numberDouble" in value:
                return float(value["$numberDouble"])
            if "$numberInt" in value:
                return int(value["$numberInt"])
            if "$numberLong" in value:
                return int(value["$numberLong"])
        return value or 0  # Return the value itself if not a dict, or 0 if None/falsy

    try:
        if not candidate_places:
            return []

        scaler = MinMaxScaler()

        for place in candidate_places:
            interactions_count = interactions_collection.count_documents(
                {"user_id": user_id, "place_id": place["_id"]}
            )

            # Extract numeric values correctly
            avg_rating = extract_number(place.get("average_rating", 0))
            likes = extract_number(place.get("likes", 0))

            # Calculate raw score using weights:
            # - Rating: 40%
            # - Likes: 30%
            # - User interactions: 30%
            place["score"] = (
                0.4 * avg_rating +  # Rating weight
                0.3 * likes / 10000 +  # Normalize likes (assume 10K is max)
                0.3 * interactions_count  # User interaction weight
            )

        # Normalize scores using MinMaxScaler
        scores = [[p["score"]] for p in candidate_places]
        if scores:
            try:
                scaled_scores = scaler.fit_transform(scores)
                for i, place in enumerate(candidate_places):
                    place["final_score"] = float(scaled_scores[i][0])  # Convert numpy type to float
            except Exception as e:
                logger.error(f"Error scaling scores: {e}")
                # Fallback to unscaled scores
                for place in candidate_places:
                    place["final_score"] = place["score"]
        else:
            for place in candidate_places:
                place["final_score"] = 0  # Default score if no data available

        # Use final_score for sorting, not the objects themselves
        return sorted(candidate_places, key=lambda x: x.get("final_score", 0), reverse=True)
    except Exception as e:
        logger.error(f"Error ranking places: {e}")
        # Use average_rating for fallback sorting, but extract numeric value first
        return sorted(candidate_places,
                     key=lambda x: extract_number(x.get("average_rating", 0)),
                     reverse=True)  # Fallback sorting

def get_previously_shown_places(user_id):
    """
    Get a list of all place IDs previously shown to the user.
    
    Args:
        user_id: User ID
        
    Returns:
        List of place IDs
    """
    try:
        user_shown = shown_places_collection.find_one({"user_id": user_id})
        return user_shown.get("place_ids", []) if user_shown else []
    except Exception as e:
        logger.error(f"Error getting previously shown places: {e}")
        return []

def get_last_shown_places(user_id):
    """
    Get only the places shown in the most recent request.
    
    Args:
        user_id: User ID
        
    Returns:
        List of place IDs from last request
    """
    try:
        user_shown = shown_places_collection.find_one({"user_id": user_id})
        return user_shown.get("last_shown_place_ids", []) if user_shown else []
    except Exception as e:
        logger.error(f"Error getting last shown places: {e}")
        return []

def reset_user_shown_places(user_id):
    """
    Reset the tracking of places shown to a user.
    
    Args:
        user_id: User ID
        
    Returns:
        Boolean indicating success
    """
    try:
        result = shown_places_collection.delete_one({"user_id": user_id})
        deleted = result.deleted_count > 0
        logger.info(f"Reset shown places for user {user_id}, success: {deleted}")
        return deleted
    except Exception as e:
        logger.error(f"Error resetting shown places: {e}")
        return False

def update_shown_places(user_id, new_place_ids, max_places=100):
    """
    Update the list of shown places for a user.
    Keeps track of places in chronological order, with most recent at the END of the list.
    
    Args:
        user_id: User ID
        new_place_ids: List of new place IDs to add to shown list
        max_places: Maximum number of places to keep in history
    """
    try:
        # Get current shown places
        shown_doc = shown_places_collection.find_one({"user_id": user_id})
        
        if shown_doc:
            # Get existing place IDs
            existing_ids = shown_doc.get("place_ids", [])
            
            # Remove new IDs if they already exist (to avoid duplicates)
            existing_ids = [pid for pid in existing_ids if pid not in new_place_ids]
            
            # Add new IDs at the END of the list (most recent)
            updated_ids = existing_ids + new_place_ids
            
            # Keep only the most recent max_places
            if len(updated_ids) > max_places:
                updated_ids = updated_ids[-max_places:]
            
            # Update document with new list and last shown
            shown_places_collection.update_one(
                {"user_id": user_id},
                {"$set": {
                    "place_ids": updated_ids, 
                    "last_shown_place_ids": new_place_ids,
                    "last_updated": datetime.now()
                }}
            )
        else:
            # Create new document
            shown_places_collection.insert_one({
                "user_id": user_id,
                "place_ids": new_place_ids,
                "last_shown_place_ids": new_place_ids,
                "last_updated": datetime.now()
            })
            
        logger.info(f"Updated shown places for user {user_id}, added {len(new_place_ids)} places")
    except Exception as e:
        logger.error(f"Error updating shown places: {str(e)}")
def get_unshown_places(user_id, limit=10):
    """
    Get places that haven't been shown to the user yet.
    
    Args:
        user_id: User ID
        limit: Maximum number of places to return
        
    Returns:
        List of place documents
    """
    try:
        # Get previously shown places
        shown_place_ids = get_previously_shown_places(user_id)

        # Get all places that haven't been shown to this user
        if shown_place_ids:
            unshown_places = list(places_collection.find({"_id": {"$nin": shown_place_ids}}).limit(limit))
        else:
            # If no shown places, get any places
            unshown_places = list(places_collection.find().limit(limit))

        return unshown_places
    except Exception as e:
        logger.error(f"Error getting unshown places: {e}")
        return []

def refresh_shown_places(user_id, shown_place_ids, limit=10):
    """
    Re-rank and refresh previously shown places based on recent activity.
    
    Args:
        user_id: User ID
        shown_place_ids: List of place IDs to refresh
        limit: Maximum number of places to return
        
    Returns:
        List of refreshed place documents
    """
    try:
        if not shown_place_ids:
            return []

        # Get the place documents
        shown_places = list(places_collection.find({"_id": {"$in": shown_place_ids}}))

        if not shown_places:
            return []

        # Get recent interaction data for all users
        recent_date = datetime.now() - timedelta(days=7)

        # Count recent interactions for each place
        place_interaction_counts = {}
        for place_id in shown_place_ids:
            count = interactions_collection.count_documents({
                "place_id": place_id,
                "timestamp": {"$gte": recent_date},
                "interaction_type": {"$in": ["like", "save", "share", "view"]}
            })
            place_interaction_counts[place_id] = count

        # Add recency score to places
        for place in shown_places:
            place["recency_score"] = place_interaction_counts.get(place["_id"], 0)

        # Sort by recency score and add some randomness for variety
        refreshed_places = sorted(
            shown_places, 
            key=lambda x: x.get("recency_score", 0) + random.random(), 
            reverse=True
        )

        return refreshed_places[:limit]
    except Exception as e:
        logger.error(f"Error refreshing shown places: {e}")
        return []
# --- PART 4: RECOMMENDATION GENERATION AND CACHING ---
def generate_final_recommendations(user_id, num_recommendations=10, previously_shown_ids=None):
    """
    Generate final personalized recommendations for a user.
    Enhanced with collaborative filtering (40% of recommendations) and explicit tracking of shown places.
    Implements advanced fallback mechanisms with randomized mixing when running out of new places.
    
    Args:
        user_id: User ID
        num_recommendations: Number of recommendations to generate
        previously_shown_ids: List of previously shown place IDs to exclude
        
    Returns:
        List of personalized recommendations
    """
    # Helper function to safely extract numeric values from MongoDB dictionary representations
    def extract_numeric(value, default=0):
        if isinstance(value, dict):
            # Handle MongoDB numeric types
            if "$numberDouble" in value:
                return float(value["$numberDouble"])
            if "$numberInt" in value:
                return float(int(value["$numberInt"]))
            if "$numberLong" in value:
                return float(int(value["$numberLong"]))
            return default
        
        # Handle direct numeric types
        if isinstance(value, (int, float)):
            return float(value)
        
        # Try converting string to float
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
            
    try:
        logger.info(f"Generating final recommendations for user {user_id}, need {num_recommendations}")
        
        # Get user preferences
        user_prefs = get_user_travel_preferences(user_id)
        
        # Use provided previously_shown_ids or fetch them if not provided
        if previously_shown_ids is None:
            previously_shown = shown_places_collection.find_one({"user_id": user_id})
            previously_shown_ids = previously_shown.get("place_ids", []) if previously_shown else []
        
        # Initialize recommendations
        recommendations = []
        
        # PART 1: Calculate recommendation distribution (40% collaborative, 60% content-based)
        collab_count = int(num_recommendations * 0.4)  # 40% for collaborative
        content_count = num_recommendations - collab_count  # 60% for content-based
        
        # PART 2: Get collaborative recommendations first (40%)
        if collab_count > 0:
            collab_places_ids = get_collaborative_recommendations(user_id)
            
            # Log the total number of collaborative recommendations found
            logger.info(f"Found {len(collab_places_ids)} collaborative recommendations for user {user_id}")
            
            # Filter out previously shown places
            collab_places_ids = [pid for pid in collab_places_ids if pid not in previously_shown_ids]
            
            # Get place details for remaining collaborative recommendations
            if collab_places_ids:
                # Get all collaborative place details
                all_collab_places = list(places_collection.find({"_id": {"$in": collab_places_ids}}))
                
                # Sort collaborative recommendations by rating for consistency - FIXED
                all_collab_places.sort(key=lambda x: extract_numeric(x.get("average_rating", 0)), reverse=True)
                
                # Use top collaborative places up to our limit
                collab_to_add = all_collab_places[:collab_count]
                
                # Add collaborative places
                for place in collab_to_add:
                    place["source"] = "collaborative"
                    recommendations.append(place)
                
                logger.info(f"Added {len(collab_to_add)} collaborative filtering recommendations")
            else:
                logger.info("No new collaborative recommendations available")
        
        # PART 3: Get content-based recommendations (60% or more if collaborative failed)
        remaining_content_count = num_recommendations - len(recommendations)
        
        if remaining_content_count > 0:
            # Get candidate places
            candidate_places = get_candidate_places(user_prefs, user_id, size=remaining_content_count * 5)
            
            # Filter out previously shown places and already added recommendations
            filtered_candidates = []
            for place in candidate_places:
                if place["_id"] not in previously_shown_ids and place["_id"] not in [r["_id"] for r in recommendations]:
                    filtered_candidates.append(place)
            
            # Apply personalization factors to remaining candidates
            ranked_places = []
            
            for place in filtered_candidates:
                # Calculate personalization score
                score = calculate_personalization_score(place, user_id, user_prefs)
                ranked_places.append((place, score))
            
            # Define helper function to safely extract score values
            def get_score(item):
                score = item[1]
                if isinstance(score, dict):
                    # Handle MongoDB numeric types
                    if "$numberDouble" in score:
                        return float(score["$numberDouble"])
                    if "$numberInt" in score:
                        return float(int(score["$numberInt"]))
                    if "$numberLong" in score:
                        return float(int(score["$numberLong"]))
                    return 0.0  # Default if it's an unrecognized dict
                try:
                    return float(score)
                except (TypeError, ValueError):
                    return 0.0

            # Sort places by personalization score using the helper function
            ranked_places.sort(key=get_score, reverse=True)
            
            # Add top content-based places
            added_content_places = 0
            for place, _ in ranked_places:
                if len(recommendations) < num_recommendations:
                    place["source"] = "content_based"
                    recommendations.append(place)
                    added_content_places += 1
                else:
                    break
            
            logger.info(f"Added {added_content_places} content-based places")
        
        # PART 4: FALLBACK OPTIONS - Prepare all potential fallback sources
        # If we still need more recommendations, prepare fallback options
        if len(recommendations) < num_recommendations:
            remaining_needed = num_recommendations - len(recommendations)
            logger.info(f"Need {remaining_needed} more recommendations, preparing fallback options")
            
            # Keep track of IDs we've already included
            current_rec_ids = [r["_id"] for r in recommendations]
            
            # FALLBACK SOURCE 1: Discovery places
            discovery_places = []
            try:
                raw_discovery = get_discovery_places(user_id, limit=remaining_needed * 2)
                for place in raw_discovery:
                    if place["_id"] not in previously_shown_ids and place["_id"] not in current_rec_ids:
                        place["source"] = "discovery"
                        discovery_places.append(place)
            except Exception as e:
                logger.error(f"Error getting discovery places: {str(e)}")
            
            # FALLBACK SOURCE 2: Trending places
            trending_places = []
            try:
                # Get recent interactions to find trending places
                recent_date = datetime.now() - timedelta(days=7)
                trending_interactions = list(interactions_collection.aggregate([
                    {"$match": {"timestamp": {"$gte": recent_date}}},
                    {"$group": {"_id": "$place_id", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                    {"$limit": remaining_needed * 2}
                ]))
                
                trending_place_ids = [item["_id"] for item in trending_interactions]
                trending_place_ids = [pid for pid in trending_place_ids 
                                    if pid not in previously_shown_ids and 
                                       pid not in current_rec_ids]
                
                if trending_place_ids:
                    raw_trending = list(places_collection.find({"_id": {"$in": trending_place_ids}}))
                    for place in raw_trending:
                        place["source"] = "trending"
                        trending_places.append(place)
            except Exception as e:
                logger.error(f"Error getting trending places: {str(e)}")
            
            # FALLBACK SOURCE 3: Old previously shown places (user likely forgot)
            old_shown_places = []
            try:
                if previously_shown_ids and len(previously_shown_ids) > 20:
                    # Get places from beginning of history (oldest ones user likely forgot)
                    # Use the first 30% of the history
                    oldest_count = min(int(len(previously_shown_ids) * 0.3), remaining_needed * 2)
                    oldest_ids = previously_shown_ids[:oldest_count]
                    
                    # Filter out IDs already in recommendations
                    oldest_ids = [pid for pid in oldest_ids if pid not in current_rec_ids]
                    
                    if oldest_ids:
                        raw_old_places = list(places_collection.find({"_id": {"$in": oldest_ids}}))
                        for place in raw_old_places:
                            place["source"] = "rediscovery"
                            old_shown_places.append(place)
            except Exception as e:
                logger.error(f"Error getting old shown places: {str(e)}")
            
            # FALLBACK SOURCE 4: Any top-rated places as last resort
            top_rated_places = []
            try:
                # Get top-rated places not already included - FIXED avoid server-side sorting
                raw_top_rated = list(places_collection.find(
                    {"_id": {"$nin": current_rec_ids + previously_shown_ids}}
                ).limit(remaining_needed * 3))  # Get more than needed
                
                # Sort client-side using extract_numeric - FIXED
                raw_top_rated.sort(key=lambda p: extract_numeric(p.get("average_rating", 0)), reverse=True)
                # Take only what we need
                raw_top_rated = raw_top_rated[:remaining_needed]
                
                for place in raw_top_rated:
                    place["source"] = "top_rated"
                    top_rated_places.append(place)
            except Exception as e:
                logger.error(f"Error getting top-rated places: {str(e)}")
            
            # PART 5: COMBINE FALLBACK SOURCES - Mix randomly to avoid patterns
            # Combine all fallback options
            all_fallbacks = discovery_places + trending_places + old_shown_places + top_rated_places
            
            # Shuffle to randomize selection (avoid sequential patterns)
            random.shuffle(all_fallbacks)
            
            # Remove any duplicates by ID
            seen_ids = set(current_rec_ids)
            unique_fallbacks = []
            for place in all_fallbacks:
                if place["_id"] not in seen_ids:
                    seen_ids.add(place["_id"])
                    unique_fallbacks.append(place)
            
            # Add fallbacks until we reach target count
            added_fallback_count = 0
            for place in unique_fallbacks:
                if len(recommendations) < num_recommendations:
                    recommendations.append(place)
                    added_fallback_count += 1
                else:
                    break
            
            logger.info(f"Added {added_fallback_count} mixed fallback recommendations")
        
        # PART 6: ABSOLUTE LAST RESORT - Reuse any places if we still need more
        if len(recommendations) < num_recommendations:
            remaining_needed = num_recommendations - len(recommendations)
            logger.info(f"Still need {remaining_needed} more places, using any available places")
            
            # Get any places excluding what we've already added - FIXED avoid server-side sorting
            current_rec_ids = [r["_id"] for r in recommendations]
            last_resort_places = list(places_collection.find(
                {"_id": {"$nin": current_rec_ids}}
            ).limit(remaining_needed * 3))  # Get more than needed
            
            # Sort client-side using extract_numeric - FIXED
            last_resort_places.sort(key=lambda p: extract_numeric(p.get("average_rating", 0)), reverse=True)
            # Take only what we need
            last_resort_places = last_resort_places[:remaining_needed]
            
            for place in last_resort_places:
                place["source"] = "last_resort"
                recommendations.append(place)
            
            logger.info(f"Added {len(last_resort_places)} last resort places")
        
        logger.info(f"Final recommendation count: {len(recommendations)}/{num_recommendations}")
        return recommendations[:num_recommendations]
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        # Return some fallback recommendations in case of error - FIXED avoid server-side sorting
        try:
            fallback_places = list(places_collection.find().limit(num_recommendations * 3))
            
            # Sort client-side using extract_numeric - FIXED
            fallback_places.sort(key=lambda p: extract_numeric(p.get("average_rating", 0)), reverse=True)
            # Take only what we need
            fallback_places = fallback_places[:num_recommendations]
            
            for place in fallback_places:
                place["source"] = "error_fallback"
            return fallback_places
        except Exception as fallback_error:
            logger.error(f"Error in fallback recommendation retrieval: {fallback_error}")
            return []  # Return empty list as ultimate fallback
def get_recommendations_with_caching(user_id, force_refresh=False, num_new_recommendations=10, max_total=30):
    """
    Get recommendations for a user with caching and progressive pagination.
    First request: Return 10 new places
    Second request: Return 10 new places + previous 10 = 20 total
    Third request and beyond: Return 10 new places + 20 most recent shown places = 30 total
    
    Args:
        user_id: User ID
        force_refresh: Whether to force generation of new recommendations
        num_new_recommendations: Number of new recommendations to return (default 10)
        max_total: Maximum total recommendations to return (default 30)
        
    Returns:
        Dict with new recommendations and previously shown recommendations
    """
    try:
        # Get previously shown places
        previously_shown = shown_places_collection.find_one({"user_id": user_id})
        previously_shown_ids = previously_shown.get("place_ids", []) if previously_shown else []
        
        # Determine how many previously shown places to return based on pagination
        total_shown_count = len(previously_shown_ids)
        
        # Progressive pagination logic:
        # - First request: Return 10 new places only (0 previous)
        # - Second request: Return 10 new + 10 previous = 20 total
        # - Third+ request: Return 10 new + 20 previous = 30 total
        if total_shown_count == 0:
            # First request - no previous places
            history_count = 0
        elif total_shown_count <= 10:
            # Second request - include up to 10 previous places
            history_count = total_shown_count
        else:
            # Third+ request - include up to 20 previous places
            history_count = min(20, total_shown_count)
        
        logger.info(f"User {user_id} pagination: {total_shown_count} total shown, returning {history_count} history items")
        
        # Get new recommendations (either from cache or generated)
        new_recommendations = []
        
        if force_refresh:
            logger.info(f"Force refresh requested for user {user_id}")
            new_recommendations = generate_final_recommendations(user_id, num_new_recommendations, previously_shown_ids)
        else:
            # Look for cached recommendations
            cached_entries = get_user_cached_recommendations(user_id)
            
            if not cached_entries:
                logger.info(f"No cached recommendations found for user {user_id}")
                new_recommendations = generate_final_recommendations(user_id, num_new_recommendations, previously_shown_ids)
            else:
                # Get the first entry with lowest sequence number
                cached_entry = cached_entries[0]
                
                # Remove the used entry from cache
                recommendations_cache_collection.delete_one({"_id": cached_entry["_id"]})
                
                logger.info(f"Using cached recommendations for user {user_id} (sequence {cached_entry['sequence']})")
                
                # If this was the last entry, scheduling will be handled by the endpoint
                if len(cached_entries) <= 2:
                    logger.info(f"Only {len(cached_entries)} cached entries left for user {user_id}, scheduling more")
                
                # Filter cached recommendations to ensure they haven't been shown before
                filtered_cache = []
                for place in cached_entry["recommendations"]:
                    if place["_id"] not in previously_shown_ids:
                        filtered_cache.append(place)
                
                # If we filtered too many, generate new ones
                if len(filtered_cache) >= num_new_recommendations:
                    new_recommendations = filtered_cache[:num_new_recommendations]
                else:
                    # Not enough from cache after filtering, generate new ones
                    logger.info(f"Only {len(filtered_cache)} new places in cache after filtering, generating more")
                    additional_needed = num_new_recommendations - len(filtered_cache)
                    
                    # Add what we have from cache
                    new_recommendations = filtered_cache.copy()
                    
                    # Generate more to supplement
                    existing_ids = [p["_id"] for p in new_recommendations]
                    additional_recs = generate_final_recommendations(
                        user_id, 
                        additional_needed, 
                        previously_shown_ids + existing_ids
                    )
                    
                    # Add the additional recommendations
                    new_recommendations.extend(additional_recs)
        
        # Get the IDs of new recommendations
        new_place_ids = [p["_id"] for p in new_recommendations]
        
        # Get previously shown places for history display (most recent N, excluding new places)
        previously_shown_places = []
        if history_count > 0:
            # Filter out places we're showing as new
            shown_ids_to_fetch = [pid for pid in previously_shown_ids if pid not in new_place_ids]
            
            # Get the most recent ones based on history_count (from end of list)
            if shown_ids_to_fetch:
                # Get the most recent items (last N items in the list)
                shown_ids_to_fetch = shown_ids_to_fetch[-history_count:]
                
                # Fetch the actual place data
                if shown_ids_to_fetch:
                    previously_shown_places = list(places_collection.find({"_id": {"$in": shown_ids_to_fetch}}))
                    
                    # Add source information
                    for place in previously_shown_places:
                        place["source"] = "history"
        
        # Track these new recommendations as shown
        update_shown_places(user_id, new_place_ids, max_places=100)
        
        return {
            "new_recommendations": new_recommendations,
            "previously_shown": previously_shown_places
        }
            
    except Exception as e:
        logger.error(f"Error getting recommendations with caching: {str(e)}")
        # Fallback to generating without cache
        new_recommendations = generate_final_recommendations(user_id, num_new_recommendations, [])
        return {
            "new_recommendations": new_recommendations,
            "previously_shown": []
        }
async def background_cache_recommendations(user_id, num_entries=6):
    """
    Generate and cache recommendation entries for a user in the background.
    Uses lazy loading to generate recommendations in batches.
    
    Args:
        user_id: User ID
        num_entries: Number of cache entries to generate
    """
    # Create a unique lock key for this user
    cache_lock_key = f"cache_lock_{user_id}"
    
    try:
        # Check if cache generation is already in progress
        lock = cache_locks_collection.find_one({"_id": cache_lock_key})
        
        if lock:
            # Check if lock is stale (older than 10 minutes)
            lock_time = lock.get("timestamp", datetime.min)
            if isinstance(lock_time, str):
                try:
                    lock_time = datetime.fromisoformat(lock_time.replace('Z', '+00:00'))
                except:
                    lock_time = datetime.min
            
            if (datetime.now() - lock_time).total_seconds() < 600:  # 10 minutes
                logger.info(f"Cache generation already in progress for user {user_id}")
                return
            else:
                logger.info(f"Found stale lock for user {user_id}, removing")
                cache_locks_collection.delete_one({"_id": cache_lock_key})
        
        # Create lock
        cache_locks_collection.insert_one({
            "_id": cache_lock_key,
            "timestamp": datetime.now(),
            "user_id": user_id
        })
        
        # Get previously shown places to exclude from cache generation
        previously_shown = shown_places_collection.find_one({"user_id": user_id})
        previously_shown_ids = previously_shown.get("place_ids", []) if previously_shown else []
        
        # Determine next sequence number
        try:
            existing_cache = list(recommendations_cache_collection.find(
                {"user_id": user_id}
            ).sort("sequence", 1))
            
            if existing_cache:
                next_sequence = max([entry.get("sequence", 0) for entry in existing_cache]) + 1
            else:
                next_sequence = 0
                
        except Exception as e:
            logger.error(f"Error getting existing cache entries: {e}")
            next_sequence = 0
            
        # Generate recommendations in sequence
        logger.info(f"Generating {num_entries} cache entries for user {user_id}")
        
        # Initialize tracking variables for lazy loading
        entries_created = 0
        batch_size = 20  # Start with smaller batch sizes
        recommendations_pool = []
        used_place_ids = set(previously_shown_ids)  # Track used places
        
        # Generate cache entries one by one using lazy loading
        for i in range(num_entries):
            if entries_created >= num_entries:
                break
            
            # Enhanced yield point - check if high priority tasks are running
            # This helps give priority to interactive requests
            high_priority_count = task_manager.running_tasks[TaskPriority.HIGH]
            if high_priority_count > 0:
                # Longer delay if high priority tasks are running
                logger.info(f"Yielding to {high_priority_count} high priority tasks")
                await asyncio.sleep(1.0)  # Longer yield when HIGH priority tasks are running
            else:
                # Short delay between iterations to allow other tasks to run
                await asyncio.sleep(0.1)
                
            try:
                # Check if we need more recommendations
                if len(recommendations_pool) < 10:
                    # Generate a new batch
                    logger.info(f"Generating batch of {batch_size} recommendations for pool")
                    
                    # Generate with slightly different weights for variety
                    randomization_seed = next_sequence + i + int(datetime.now().timestamp())
                    random.seed(randomization_seed)
                    
                    # Generate batch of new recommendations excluding shown and used places
                    exclude_ids = list(used_place_ids)
                    batch = generate_final_recommendations(user_id, batch_size, exclude_ids)
                    
                    # Add to pool and mark as used
                    for rec in batch:
                        if rec["_id"] not in used_place_ids:
                            recommendations_pool.append(rec)
                            used_place_ids.add(rec["_id"])
                    
                    # If we got less than expected, increase batch size for next round
                    if len(batch) < batch_size:
                        batch_size = min(batch_size * 2, 50)  # Double but cap at 50
                    
                    # Ensure we have variety by shuffling
                    random.shuffle(recommendations_pool)
                
                # If we still don't have enough recommendations, try fallback approach
                if len(recommendations_pool) < 10:
                    # Generate fallback recommendations
                    fallback_count = 10 - len(recommendations_pool)
                    existing_rec_ids = [r["_id"] for r in recommendations_pool]
                    
                    # Generate new recommendations excluding both previously shown and those already in our pool
                    exclude_ids = list(used_place_ids)
                    fallbacks = generate_final_recommendations(user_id, fallback_count, exclude_ids)
                    
                    # Add fallbacks to pool
                    for rec in fallbacks:
                        if rec["_id"] not in used_place_ids:
                            recommendations_pool.append(rec)
                            used_place_ids.add(rec["_id"])
                
                # If we still don't have enough, continue to next iteration
                if len(recommendations_pool) < 10:
                    logger.warning(f"Not enough recommendations for cache entry {i+1}, got {len(recommendations_pool)}")
                    continue
                
                # Take 10 recommendations from the pool
                recommendations = recommendations_pool[:10]
                recommendations_pool = recommendations_pool[10:]
                
                # Store in cache with incrementing sequence
                sequence = next_sequence + i
                
                # Store with generation parameters for debugging
                recommendations_cache_collection.insert_one({
                    "user_id": user_id,
                    "sequence": sequence,
                    "recommendations": recommendations,
                    "timestamp": datetime.now(),
                    "generation_params": {
                        "randomization_seed": randomization_seed,
                        "excluded_shown_count": len(previously_shown_ids),
                        "lazy_loading": True
                    }
                })
                
                entries_created += 1
                logger.info(f"Generated cache entry {entries_created}/{num_entries} for user {user_id} (sequence {sequence})")
                
                # Add a delay between generations for variety - enhanced to check task priority
                delay = 0.5
                if task_manager.running_tasks[TaskPriority.HIGH] > 0:
                    # Longer delay if high priority tasks are running
                    delay = 1.0
                await asyncio.sleep(delay)
                
            except Exception as entry_error:
                logger.error(f"Error generating cache entry {i+1}/{num_entries}: {entry_error}")
                
    except Exception as e:
        logger.error(f"Error in background cache task: {e}")
        
    finally:
        # Always release the lock
        try:
            cache_locks_collection.delete_one({"_id": cache_lock_key})
            logger.info(f"Released cache lock for user {user_id}")
        except Exception as lock_error:
            logger.error(f"Error releasing cache lock: {lock_error}")
# --- PART 5: ROADMAP GENERATION ---future enhancements

def get_seasonal_activities(month=None):
    """
    Get seasonal activities based on the current month or specified month.
    
    Args:
        month: Month name (e.g. "January"), defaults to current month
        
    Returns:
        Dictionary mapping seasons to weights
    """
    if not month:
        # Default to current month
        month = datetime.now().strftime("%B")
    
    # Map months to seasons with weights
    season_weights = {
        "winter": 0,
        "spring": 0,
        "summer": 0,
        "fall": 0
    }
    
    # Northern hemisphere seasons
    if month in ["December", "January", "February"]:
        season_weights["winter"] = 1.0
        season_weights["fall"] = 0.3
    elif month in ["March", "April", "May"]:
        season_weights["spring"] = 1.0
        season_weights["winter"] = 0.3
    elif month in ["June", "July", "August"]:
        season_weights["summer"] = 1.0
        season_weights["spring"] = 0.3
    elif month in ["September", "October", "November"]:
        season_weights["fall"] = 1.0
        season_weights["summer"] = 0.3
    
    return season_weights

def get_budget_mappings():
    """
    Get budget level mappings.
    
    Returns:
        Dictionary of budget levels and their numerical mappings
    """
    return {
        "budget": 1,
        "low": 1,
        "economy": 1,
        "mid-range": 2,
        "medium": 2,
        "moderate": 2,
        "high-end": 3,
        "high": 3,
        "luxury": 4,
        "premium": 4,
        "exclusive": 5
    }

def map_budget_level(budget_text):
    """
    Map a text budget description to a numerical level (1-5).
    
    Args:
        budget_text: Text description of budget
        
    Returns:
        Numerical budget level from 1 (lowest) to 5 (highest)
    """
    if not budget_text:
        return 3  # Default to medium
    
    budget_text = budget_text.lower().strip()
    budget_mappings = get_budget_mappings()
    
    # Direct match
    if budget_text in budget_mappings:
        return budget_mappings[budget_text]
    
    # Substring match
    for key, value in budget_mappings.items():
        if key in budget_text:
            return value
    
    # Default to medium if no match
    return 3

def calculate_budget_compatibility(place_budget_level, user_budget_level):
    """
    Calculate compatibility score between place budget and user budget.
    
    Args:
        place_budget_level: Numerical budget level of place (1-5)
        user_budget_level: Numerical budget level of user (1-5)
        
    Returns:
        Compatibility score between 0 and 1
    """
    # Helper function to safely extract numeric values
    def ensure_numeric(value, default=3):
        if isinstance(value, dict):
            # Handle MongoDB numeric types
            if "$numberDouble" in value:
                return float(value["$numberDouble"])
            if "$numberInt" in value:
                return int(value["$numberInt"])
            if "$numberLong" in value:
                return int(value["$numberLong"])
            return default  # Default if it's an unrecognized dict
        try:
            return float(value) if isinstance(value, str) else value
        except (ValueError, TypeError):
            return default
            
    # Standardize budget levels
    if isinstance(place_budget_level, str):
        place_budget_level = map_budget_level(place_budget_level)
    else:
        # Handle potential MongoDB dictionary representation
        place_budget_level = ensure_numeric(place_budget_level)
        
    if isinstance(user_budget_level, str):
        user_budget_level = map_budget_level(user_budget_level)
    else:
        # Handle potential MongoDB dictionary representation
        user_budget_level = ensure_numeric(user_budget_level)
    
    # Calculate the absolute difference (0-4)
    diff = abs(place_budget_level - user_budget_level)
    
    # Convert to 0-1 score (1 is perfect match, 0 is worst match)
    # Maximum difference is 4, so we do 1 - (diff/4)
    score = 1 - (diff / 4)
    
    return score
def check_accessibility_compatibility(place, accessibility_needs):
    """
    Check if a place is compatible with the user's accessibility needs.
    
    Args:
        place: Place document
        accessibility_needs: List of accessibility requirements
        
    Returns:
        Boolean indicating whether the place meets all accessibility requirements
    """
    if not accessibility_needs:
        return True  # No requirements, so all places are compatible
    
    # Get place accessibility features - FIXED: changed field name to match data structure
    place_features = place.get("accessibility", [])
    
    # Check if all required features are present
    for need in accessibility_needs:
        if need not in place_features:
            return False
    
    return True

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points using the Haversine formula.
    
    Args:
        lat1: Latitude of point 1
        lon1: Longitude of point 1
        lat2: Latitude of point 2
        lon2: Longitude of point 2
        
    Returns:
        Distance in kilometers
    """
    # Helper function to safely extract a numeric value
    def ensure_float(value):
        if isinstance(value, dict):
            # Handle MongoDB numeric types
            if "$numberDouble" in value:
                return float(value["$numberDouble"])
            if "$numberInt" in value:
                return float(int(value["$numberInt"]))
            if "$numberLong" in value:
                return float(int(value["$numberLong"]))
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    
    # Safely convert all coordinates to floats
    lat1 = ensure_float(lat1)
    lon1 = ensure_float(lon1)
    lat2 = ensure_float(lat2)
    lon2 = ensure_float(lon2)
    
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def generate_routes(places):
    """
    Generate routes between places in a roadmap.
    
    Args:
        places: List of places
        
    Returns:
        List of route objects
    """
    routes = []
    
    # Helper function to safely extract a numeric value
    def ensure_float(value):
        if isinstance(value, dict):
            # Handle MongoDB numeric types
            if "$numberDouble" in value:
                return float(value["$numberDouble"])
            if "$numberInt" in value:
                return float(int(value["$numberInt"]))
            if "$numberLong" in value:
                return float(int(value["$numberLong"]))
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    
    # For MVP, just create sequential routes between places
    for i in range(len(places) - 1):
        from_place = places[i]
        to_place = places[i+1]
        
        # Calculate distance between places if lat/long available
        distance = None
        from_loc = from_place.get("location", {})
        to_loc = to_place.get("location", {})
        
        if "latitude" in from_loc and "longitude" in from_loc and "latitude" in to_loc and "longitude" in to_loc:
            try:
                # Safely convert coordinates to float
                from_lat = ensure_float(from_loc["latitude"])
                from_lon = ensure_float(from_loc["longitude"])
                to_lat = ensure_float(to_loc["latitude"])
                to_lon = ensure_float(to_loc["longitude"])
                
                distance = calculate_distance(from_lat, from_lon, to_lat, to_lon)
                # Round to 1 decimal place
                distance = round(distance, 1)
            except Exception as e:
                logger.warning(f"Error calculating distance: {e}")
        
        routes.append({
            "from": from_place["_id"],
            "to": to_place["_id"],
            "distance_km": distance,
            "from_name": from_place.get("name", "Unknown"),
            "to_name": to_place.get("name", "Unknown")
        })
    
    return routes
from collections import Counter
def generate_hybrid_roadmap(user_id):
    """
    Generate a travel roadmap for a user using a hybrid approach with mixed recommendation strategies.
    Prioritizes saved places and places in the user's chosen destinations across all recommendation categories.
    Only looks outside destinations if there aren't enough matches.
    Always returns exactly 10 places using a psychologically-optimized balance.
    
    Args:
        user_id: User ID
        
    Returns:
        Dictionary containing roadmap data with exactly 10 places
    """

    
    """
    Generate a travel roadmap for a user using a hybrid approach with mixed recommendation strategies.
    ...
    """
    # Helper function to safely extract numeric values from MongoDB data
    def extract_numeric(value, default=0):
        if isinstance(value, dict):
            # Handle MongoDB numeric types
            if "$numberDouble" in value:
                return float(value["$numberDouble"])
            if "$numberInt" in value:
                return float(int(value["$numberInt"]))
            if "$numberLong" in value:
                return float(int(value["$numberLong"]))
            return default
        
        # Handle direct numeric types
        if isinstance(value, (int, float)):
            return float(value)
        
        # Try converting string to float
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    logger.info(f"Generating roadmap for user {user_id}")
    
    # Get user info
    user = users_collection.find_one({"_id": user_id})
    if not user:
        logger.warning(f"User {user_id} not found")
        return {"error": "User not found"}
    
    # Get user travel preferences
    travel_prefs = get_user_travel_preferences(user_id)
    
    if not travel_prefs:
        logger.warning(f"No travel preferences found for user {user_id}")
        return {"error": "No travel preferences found"}
    
    # Extract preferences
    budget = travel_prefs.get("budget", "medium")
    budget_level = map_budget_level(budget)
    accessibility_needs = travel_prefs.get("accessibility_needs", [])
    group_type = travel_prefs.get("group_type", "")
    travel_dates = travel_prefs.get("travel_dates", "")
    
    # Get preferred destinations (if any)
    destinations = travel_prefs.get("destinations", [])
    
    # Extract month from travel dates
    travel_month = parse_travel_dates(travel_dates)
    
    logger.info(f"User {user_id} preferences: Budget={budget}, Group={group_type}, Month={travel_month}")
    logger.info(f"Accessibility needs: {accessibility_needs}, Destinations: {destinations}")
    
    # Get all possible places
    all_places = list(places_collection.find())
    
    # Track warnings to return to the user
    roadmap_warnings = []
    
    # --- CRITICAL FILTERS PIPELINE ---
    
    # STAGE 1: Accessibility Filter
    if accessibility_needs:
        filtered_places = [
            place for place in all_places
            if check_accessibility_compatibility(place, accessibility_needs)
        ]
        
        # FALLBACK 1: If no places match accessibility, use all places
        if len(filtered_places) == 0:
            warning_msg = f"⚠️ No places match accessibility needs {accessibility_needs}, using all places"
            logger.warning(warning_msg)
            roadmap_warnings.append({
                "type": "accessibility",
                "message": f"No places found matching all your accessibility needs: {', '.join(accessibility_needs)}. We've included places that may not fully meet your requirements."
            })
            filtered_places = all_places
    else:
        filtered_places = all_places
    
    logger.info(f"After accessibility filter: {len(filtered_places)}/{len(all_places)} places remaining")
    
    # STAGE 2: Identify places in chosen destinations
    destination_places = []
    if destinations and len(destinations) > 0:
        for place in filtered_places:
            location = place.get("location", {})
            city = location.get("city", "")
            country = location.get("country", "")
            
            # Check if place matches any preferred destination
            for destination in destinations:
                if (destination.lower() in city.lower() or 
                    destination.lower() in country.lower() or
                    city.lower() in destination.lower() or
                    country.lower() in destination.lower()):
                    destination_places.append(place)
                    break
        
        logger.info(f"After destination filter: {len(destination_places)} places in requested destinations")
        
        # Add warning if no places in destinations
        if len(destination_places) == 0:
            warning_msg = f"⚠️ No places found in requested destinations {destinations}"
            logger.warning(warning_msg)
            roadmap_warnings.append({
                "type": "destination",
                "message": f"No places found in your requested destinations: {', '.join(destinations)}. We'll include places from nearby or popular alternatives."
            })
    else:
        # If no destinations specified, all filtered places are potential candidates
        destination_places = filtered_places
    
    # --- NEW SECTION: PRIORITIZE SAVED PLACES ---
    # Track already selected places
    selected_place_ids = set()
    mixed_recommendations = []
    
    # Check if user has saved places
    saved_place_ids = user.get("saved_places", [])
    if saved_place_ids:
        logger.info(f"User {user_id} has {len(saved_place_ids)} saved places")
        
        # Get saved places that match our filters
        saved_places_in_destinations = []
        
        # If destinations specified, prioritize saved places in those destinations
        if destinations and len(destinations) > 0:
            for place_id in saved_place_ids:
                # Skip if already selected
                if place_id in selected_place_ids:
                    continue
                    
                # Find the place
                place = places_collection.find_one({"_id": place_id})
                if not place:
                    continue
                
                # Check if it's in the destination places
                if any(p["_id"] == place_id for p in destination_places):
                    saved_places_in_destinations.append(place)
        else:
            # No specific destinations, just check accessibility
            for place_id in saved_place_ids:
                # Skip if already selected
                if place_id in selected_place_ids:
                    continue
                    
                # Find the place
                place = places_collection.find_one({"_id": place_id})
                if not place:
                    continue
                
                # Check if it passed the accessibility filter
                if any(p["_id"] == place_id for p in filtered_places):
                    saved_places_in_destinations.append(place)
        
        # Add saved places to recommendations with high priority
        for place in saved_places_in_destinations:
            mixed_recommendations.append({
                "place": place,
                "score": 0.95,  # Very high score for saved places
                "budget_score": 0.8,
                "accessibility_score": 0.8,
                "group_score": 0.8,
                "seasonal_score": 0.8,
                "source": "saved_place"  # New source type for saved places
            })
            selected_place_ids.add(place["_id"])
            logger.info(f"Added saved place: {place.get('name')}")
    
    # --- MIXED RECOMMENDATION APPROACH ---
    # Define mix ratios based on whether destinations are specified
    if destinations and len(destinations) > 0:
        # If destinations specified, all categories focus on those destinations
        mix_ratios = {
            "destination_trending": 0.3,    # 30% - Trending places in chosen destinations
            "destination_personalized": 0.4, # 40% - Places in destinations matching user preferences
            "destination_discovery": 0.2,    # 20% - Diverse places within chosen destinations
            "destination_top": 0.1           # 10% - Top-rated places in chosen destinations
        }
    else:
        # If no destinations specified, use general categories
        mix_ratios = {
            "trending": 0.3,          # 30% - Trending places
            "personalized": 0.4,      # 40% - Places matching user preferences
            "discovery": 0.2,         # 20% - Discovery/serendipity
            "top_rated": 0.1          # 10% - Top-rated places
        }
    
    # Calculate remaining places needed after including saved places
    remaining_places_needed = 10 - len(mixed_recommendations)
    
    # Only proceed with mixed recommendations if we need more places
    if remaining_places_needed > 0:
        # Recalculate counts based on remaining places needed
        category_counts = {}
        remaining = remaining_places_needed
        for category, ratio in mix_ratios.items():
            count = max(1, int(remaining_places_needed * ratio))
            category_counts[category] = count
            remaining -= count
        
        # Distribute any remaining spots to personalized matches
        if remaining > 0:
            if destinations and len(destinations) > 0:
                category_counts["destination_personalized"] += remaining
            else:
                category_counts["personalized"] += remaining
        
        # --- 1. TRENDING PLACES IN DESTINATIONS / GENERAL TRENDING ---
        trending_category = "destination_trending" if destinations and len(destinations) > 0 else "trending"
        if category_counts.get(trending_category, 0) > 0:
            count = category_counts[trending_category]
            recent_date = datetime.now() - timedelta(days=14)
            
            if destinations and len(destinations) > 0:
                # Get trending places in destinations
                trending_pipeline = [
                    {"$match": {"timestamp": {"$gte": recent_date}}},
                    {"$lookup": {
                        "from": "places", 
                        "localField": "place_id", 
                        "foreignField": "_id", 
                        "as": "place_info"
                    }},
                    {"$unwind": "$place_info"},
                    {"$match": {"place_info.location.city": {"$in": destinations}}},
                    {"$group": {"_id": "$place_id", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                    {"$limit": count * 2}
                ]
                
                try:
                    trending_in_dest = list(interactions_collection.aggregate(trending_pipeline))
                    
                    # Add trending in destinations
                    for item in trending_in_dest:
                        if item["_id"] not in selected_place_ids:
                            place = places_collection.find_one({"_id": item["_id"]})
                            if place:
                                mixed_recommendations.append({
                                    "place": place,
                                    "score": 0.8,  # High score for trending in destination
                                    "budget_score": 0.5,
                                    "accessibility_score": 0.5,
                                    "group_score": 0.5,
                                    "seasonal_score": 0.5,
                                    "source": "trending_in_destination"
                                })
                                selected_place_ids.add(place["_id"])
                                logger.info(f"Added trending in destination: {place.get('name')}")
                except Exception as e:
                    logger.error(f"Error getting trending in destinations: {str(e)}")
            else:
                # Get general trending places (when no destinations specified)
                try:
                    general_trending = list(interactions_collection.aggregate([
                        {"$match": {"timestamp": {"$gte": recent_date}}},
                        {"$group": {"_id": "$place_id", "count": {"$sum": 1}}},
                        {"$sort": {"count": -1}},
                        {"$limit": count * 2}
                    ]))
                    
                    for item in general_trending:
                        if item["_id"] not in selected_place_ids:
                            place = places_collection.find_one({"_id": item["_id"]})
                            if place:
                                mixed_recommendations.append({
                                    "place": place,
                                    "score": 0.6,  # Medium-high score for general trending
                                    "budget_score": 0.5,
                                    "accessibility_score": 0.5,
                                    "group_score": 0.5,
                                    "seasonal_score": 0.5,
                                    "source": "trending_general"
                                })
                                selected_place_ids.add(place["_id"])
                                logger.info(f"Added general trending: {place.get('name')}")
                except Exception as e:
                    logger.error(f"Error getting general trending places: {str(e)}")
        
# --- 2. PERSONALIZED MATCHES IN DESTINATIONS / GENERAL PERSONALIZED ---
        personalized_category = "destination_personalized" if destinations and len(destinations) > 0 else "personalized"
        if category_counts.get(personalized_category, 0) > 0:
            count = category_counts[personalized_category]
            
            # Choose which places to score based on destinations
            candidate_places = destination_places if destinations and len(destinations) > 0 else filtered_places
            candidate_places = [p for p in candidate_places if p["_id"] not in selected_place_ids]
            
            # Score places based on user preference match
            scored_personalized = []
            for place in candidate_places:
                # 1. Budget score (30%)
                place_budget = place.get("budget", "medium")
                place_budget_level = map_budget_level(place_budget)
                budget_score = calculate_budget_compatibility(place_budget_level, budget_level)
                
                # 2. Accessibility score (20%)
                place_accessibility = place.get("accessibility", [])
                if not isinstance(place_accessibility, list):
                    place_accessibility = [place_accessibility] if place_accessibility else []
                
                accessibility_score = len(place_accessibility) / 10  # Normalize assuming max 10 features
                
                # 3. Group type score (30%)
                group_score = 0.5  # Default score
                place_group_type = place.get("group_type", "")
                
                if group_type and place_group_type:
                    # Handle case where place_group_type is a list
                    if isinstance(place_group_type, list):
                        # Convert list to string for comparison
                        place_group_type_str = ", ".join(place_group_type)
                    else:
                        place_group_type_str = place_group_type
                        
                    # Ensure both are strings for comparison
                    group_type_str = group_type if isinstance(group_type, str) else str(group_type)
                    
                    if group_type_str.lower() == place_group_type_str.lower():
                        group_score = 1.0  # Exact match
                    elif group_type_str.lower() in place_group_type_str.lower() or place_group_type_str.lower() in group_type_str.lower():
                        group_score = 0.8  # Partial match
                
                # 4. Seasonal score (20%)
                seasonal_score = 0.5  # Default score
                appropriate_time = place.get("appropriate_time", [])
                
                if travel_month and appropriate_time and isinstance(appropriate_time, list):
                    if travel_month in appropriate_time:
                        seasonal_score = 1.0  # Direct month match
                
                # Calculate total weighted score
                total_score = (
                    budget_score * 0.3 +
                    accessibility_score * 0.2 +
                    group_score * 0.3 +
                    seasonal_score * 0.2
                )
                
                source_type = "personalized_in_destination" if destinations and len(destinations) > 0 else "personalized_match"
                
                scored_personalized.append({
                    "place": place,
                    "score": total_score,
                    "budget_score": budget_score,
                    "accessibility_score": accessibility_score,
                    "group_score": group_score,
                    "seasonal_score": seasonal_score,
                    "source": source_type
                })
            
            # Sort by score and get top matches
            scored_personalized.sort(key=lambda x: x["score"], reverse=True)
            
            # Add top personalized matches
            for item in scored_personalized[:count]:
                if item["place"]["_id"] not in selected_place_ids:
                    mixed_recommendations.append(item)
                    selected_place_ids.add(item["place"]["_id"])
                    logger.info(f"Added personalized match: {item['place'].get('name')} (score: {item['score']:.2f})")
        
        # --- 3. DIVERSITY / DISCOVERY IN DESTINATIONS / GENERAL DISCOVERY ---
        discovery_category = "destination_discovery" if destinations and len(destinations) > 0 else "discovery"
        if category_counts.get(discovery_category, 0) > 0:
            count = category_counts[discovery_category]
            
            # Choose places from underrepresented categories
            candidate_places = destination_places if destinations and len(destinations) > 0 else filtered_places
            candidate_places = [p for p in candidate_places if p["_id"] not in selected_place_ids]
            
            # Extract categories from already selected places
            selected_categories = []
            for rec in mixed_recommendations:
                place_category = rec["place"].get("category", "").lower()
                if place_category:
                    selected_categories.append(place_category)
            
            # Group candidate places by category
            category_groups = {}
            for place in candidate_places:
                category = place.get("category", "").lower()
                if not category:
                    category = "uncategorized"
                
                if category not in category_groups:
                    category_groups[category] = []
                
                category_groups[category].append(place)
            
            # Select from least represented categories first
            diversity_picks = []
            
            # Count categories
            category_counts_map = Counter(selected_categories)
            
            # Sort categories by frequency
            sorted_categories = sorted(
                category_groups.keys(),
                key=lambda c: (category_counts_map.get(c, 0), -len(category_groups.get(c, [])))
            )
            
            # Select diversity picks
            for category in sorted_categories:
                if len(diversity_picks) >= count:
                    break
                
                places_in_cat = category_groups.get(category, [])
                if places_in_cat:
                    # Sort by rating for each category - FIXED: using average_rating with extract_numeric
                    places_in_cat.sort(key=lambda p: extract_numeric(p.get("average_rating", 0)), reverse=True)
                    
                    # Add top rated from category
                    place = places_in_cat[0]
                    source_type = "diversity_category_in_destination" if destinations and len(destinations) > 0 else "diversity_category"
                    
                    diversity_picks.append({
                        "place": place,
                        "score": 0.7,  # Medium-high score for diversity
                        "budget_score": 0.5,
                        "accessibility_score": 0.5,
                        "group_score": 0.5,
                        "seasonal_score": 0.5,
                        "source": source_type
                    })
                    
                    # Add to tracking
                    selected_place_ids.add(place["_id"])
                    selected_categories.append(category)
                    logger.info(f"Added diversity pick: {place.get('name')} (category: {category})")
            
            # Add diversity picks to recommendations
            mixed_recommendations.extend(diversity_picks)
        
       # --- 4. TOP RATED IN DESTINATIONS / GENERAL TOP RATED ---
        top_category = "destination_top" if destinations and len(destinations) > 0 else "top_rated"
        if category_counts.get(top_category, 0) > 0:
            count = category_counts[top_category]
            
            # Choose places based on top ratings
            candidate_places = destination_places if destinations and len(destinations) > 0 else filtered_places
            candidate_places = [p for p in candidate_places if p["_id"] not in selected_place_ids]
            
            # Sort by rating - use average_rating field and handle MongoDB numeric types
            candidate_places.sort(
                key=lambda p: extract_numeric(p.get("average_rating", 0)),
                reverse=True
            )
            
            # Add top rated
            added = 0
            for place in candidate_places:
                if added >= count:
                    break
                
                source_type = "top_rated_in_destination" if destinations and len(destinations) > 0 else "top_rated"
                mixed_recommendations.append({
                    "place": place,
                    "score": 0.65,  # Medium score for top rated
                    "budget_score": 0.5,
                    "accessibility_score": 0.5,
                    "group_score": 0.5,
                    "seasonal_score": 0.5,
                    "source": source_type
                })
                
                selected_place_ids.add(place["_id"])
                added += 1
                logger.info(f"Added top rated: {place.get('name')} (rating: {extract_numeric(place.get('average_rating', 'N/A'))})")  # FIXED: Added closing parenthesis
    
# --- FALLBACK MECHANISMS ---
    # If we still don't have 10 places, implement fallbacks
    if len(mixed_recommendations) < 10:
        needed = 10 - len(mixed_recommendations)
        logger.info(f"Need {needed} more places after mixed recommendations, using fallbacks")
        
        # FALLBACK 1: Look for nearby places (within 100km of destination places)
        if destinations and len(destinations) > 0 and len(destination_places) > 0:
            logger.info("Using nearby places fallback")
            
            # Find places near destination places
            nearby_places = []
            for dest_place in destination_places:
                if dest_place["_id"] in selected_place_ids:
                    continue
                
                dest_location = dest_place.get("location", {})
                if "latitude" in dest_location and "longitude" in dest_location:
                    dest_lat = dest_location["latitude"]
                    dest_lon = dest_location["longitude"]
                    
                    # Find places within 100km
                    for place in filtered_places:
                        if place["_id"] in selected_place_ids:
                            continue
                        
                        place_location = place.get("location", {})
                        if "latitude" in place_location and "longitude" in place_location:
                            place_lat = place_location["latitude"]
                            place_lon = place_location["longitude"]
                            
                            # Calculate distance
                            distance = calculate_distance(dest_lat, dest_lon, place_lat, place_lon)
                            
                            if distance <= 100:  # 100km radius
                                nearby_places.append({
                                    "place": place,
                                    "distance": distance,
                                    "score": 0.5,  # Medium score for nearby
                                    "budget_score": 0.5,
                                    "accessibility_score": 0.5,
                                    "group_score": 0.5,
                                    "seasonal_score": 0.5,
                                    "source": "nearby_to_destination"
                                })
            
            # Sort by distance
            nearby_places.sort(key=lambda x: x["distance"])
            
            # Add nearby places
            for item in nearby_places:
                if len(mixed_recommendations) >= 10:
                    break
                
                place = item["place"]
                if place["_id"] not in selected_place_ids:
                    mixed_recommendations.append(item)
                    selected_place_ids.add(place["_id"])
                    logger.info(f"Added nearby place: {place.get('name')} ({item['distance']:.1f}km)")
        
        # FALLBACK 2: Trending places (regardless of destination)
        if len(mixed_recommendations) < 10:
            logger.info("Using trending places fallback")
            
            recent_date = datetime.now() - timedelta(days=14)
            needed = 10 - len(mixed_recommendations)
            
            try:
                general_trending = list(interactions_collection.aggregate([
                    {"$match": {"timestamp": {"$gte": recent_date}}},
                    {"$group": {"_id": "$place_id", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                    {"$limit": needed * 2}
                ]))
                
                for item in general_trending:
                    if len(mixed_recommendations) >= 10:
                        break
                        
                    if item["_id"] not in selected_place_ids:
                        place = places_collection.find_one({"_id": item["_id"]})
                        if place:
                            mixed_recommendations.append({
                                "place": place,
                                "score": 0.4,  # Lower score for fallback trending
                                "budget_score": 0.5,
                                "accessibility_score": 0.5,
                                "group_score": 0.5,
                                "seasonal_score": 0.5,
                                "source": "trending_outside_destination"
                            })
                            selected_place_ids.add(place["_id"])
                            logger.info(f"Added fallback trending: {place.get('name')}")
            except Exception as e:
                logger.error(f"Error getting fallback trending places: {str(e)}")
        
        # FALLBACK 3: Top rated places (last resort)
        if len(mixed_recommendations) < 10:
            logger.info("Using top rated fallback")
            
            remaining_places = [p for p in all_places if p["_id"] not in selected_place_ids]
            # FIXED: using average_rating with extract_numeric
            remaining_places.sort(key=lambda p: extract_numeric(p.get("average_rating", 0)), reverse=True)
            
            for place in remaining_places:
                if len(mixed_recommendations) >= 10:
                    break
                    
                mixed_recommendations.append({
                    "place": place,
                    "score": 0.3,  # Low score for last resort
                    "budget_score": 0.5,
                    "accessibility_score": 0.5,
                    "group_score": 0.5,
                    "seasonal_score": 0.5,
                    "source": "fallback_outside_destination"
                })
                selected_place_ids.add(place["_id"])
                logger.info(f"Added fallback top rated: {place.get('name')}")
    
    # Ensure we have exactly 10 places (trim if more than 10)
    if len(mixed_recommendations) > 10:
        mixed_recommendations = mixed_recommendations[:10]
    
    # Sort by score for final ordering
    mixed_recommendations.sort(key=lambda x: x["score"], reverse=True)
    
    # Prepare final roadmap
    places = []
    for rec in mixed_recommendations:
        # Add match scores to the place
        place = rec["place"]
        place["match_scores"] = {
            "overall": rec["score"],
            "budget": rec["budget_score"],
            "accessibility": rec["accessibility_score"],
            "group": rec["group_score"],
            "seasonal_score": rec["seasonal_score"],
            "source": rec["source"]
        }
        places.append(place)
    
    # Build roadmap with places and routes
    roadmap = {
        "places": places,
        "routes": generate_routes(places),
        "warnings": roadmap_warnings
    }
    
    # Add a summary of warnings at the top level for API consumers
    if roadmap_warnings:
        warning_types = [w["type"] for w in roadmap_warnings]
        roadmap["has_warnings"] = True
        roadmap["warning_summary"] = f"This roadmap has {len(roadmap_warnings)} warning(s): {', '.join(warning_types)}"
    else:
        roadmap["has_warnings"] = False
    
    # Add information about saved places if any were included
    saved_places_count = sum(1 for rec in mixed_recommendations if rec.get("source") == "saved_place")
    if saved_places_count > 0:
        roadmap["saved_places_included"] = saved_places_count
        logger.info(f"Included {saved_places_count} saved places in the roadmap")
    
    logger.info(f"Generated roadmap with {len(roadmap['places'])} places and {len(roadmap['routes'])} routes")
    return roadmap
def simplify_roadmap_to_list(roadmap_data):
    """
    Simplify the roadmap data to a flat list format for easier consumption.
    
    Args:
        roadmap_data: Original roadmap data
        
    Returns:
        Dictionary with warnings/message and list of places with route information
    """
    if not roadmap_data or "places" not in roadmap_data:
        return {"data": [], "message": None}
    
    places = roadmap_data.get("places", [])
    routes = roadmap_data.get("routes", [])
    
    # Create a mapping from place ID to route
    next_stops = {}
    for route in routes:
        from_id = route.get("from")
        if from_id:
            next_stops[from_id] = route
    
    # Create the simplified list for places only
    simplified_places = []
    
    # Check if there are warnings
    has_warnings = "warnings" in roadmap_data and roadmap_data["warnings"] and len(roadmap_data["warnings"]) > 0
    
    # Prepare message object
    message = None
    if has_warnings:
        # Always include all warnings in an array
        warnings = roadmap_data["warnings"]
        warnings_list = [
            {
                "warning_type": w.get("type", "general"),
                "message": w.get("message", "Warning")
            } for w in warnings
        ]
        
        message = {
            "type": "warnings",
            "warnings": warnings_list,
            "count": len(warnings_list),
            "is_warning": True
        }
    else:
        # Add a positive message when there are no warnings
        message = {
            "type": "message",
            "message_type": "success",
            "message": "This roadmap matches your preferences perfectly!",
            "is_warning": False
        }
    
    # Then add the place entries
    for place in places:
        place_id = place.get("_id")
        
        # Get the route to the next place if available
        next_route = next_stops.get(place_id, {})
        
        # Extract and clean average_rating
        average_rating = place.get("average_rating")
        # Handle MongoDB formatted number
        if isinstance(average_rating, dict) and "$numberDouble" in average_rating:
            try:
                average_rating = float(average_rating.get("$numberDouble"))
            except (ValueError, TypeError):
                average_rating = None
        
        # Create a simplified place object matching the document structure
        simplified_place = {
            "place_id": place.get("_id"),
            "name": place.get("name"),
            "category": place.get("category"),
            "tags": place.get("tags", []),
            "description": place.get("description"),
            "location": {
                "city": place.get("location", {}).get("city"),
                "country": place.get("location", {}).get("country"),
                "latitude": place.get("location", {}).get("latitude"),
                "longitude": place.get("location", {}).get("longitude")
            },
            "accessibility": place.get("accessibility", []),  # This is now a top-level field
            "average_rating": average_rating,  # Use the cleaned value
            "likes": place.get("likes"),
            "reviews_count": place.get("reviews_count"),
            "appropriate_time": place.get("appropriate_time", []),
            "budget": place.get("budget"),
            "group_type": place.get("group_type")
        }
        
        # Create simplified entry with only place and next destination
        entry = {
            "place": simplified_place,
            "next_destination": next_route.get("to_name") if next_route else None
        }
        
        simplified_places.append(entry)
    
    # Return a structured response with message and data
    return {
        "message": message,
        "data": simplified_places
    }
async def get_roadmap_with_caching(user_id):
    """
    Get a roadmap for a user with caching.
    Only regenerates if user preferences have changed since last generation.
    
    Args:
        user_id: User ID
        
    Returns:
        Roadmap data
    """
    try:
        # Check for cached roadmap
        cached_roadmap = roadmaps_collection.find_one({"user_id": user_id})
        
        if cached_roadmap:
            logger.info(f"Found cached roadmap for user {user_id}")
            
            # Check if user preferences have changed
            current_prefs = get_user_travel_preferences(user_id)
            cached_prefs = cached_roadmap.get("travel_preferences")
            
            if current_prefs and cached_prefs:
                # Compare only the preference fields that affect roadmap generation
                preferences_changed = (
                    current_prefs.get("budget") != cached_prefs.get("budget") or
                    current_prefs.get("accessibility_needs") != cached_prefs.get("accessibility_needs") or
                    current_prefs.get("group_type") != cached_prefs.get("group_type") or
                    current_prefs.get("travel_dates") != cached_prefs.get("travel_dates") or
                    current_prefs.get("destinations") != cached_prefs.get("destinations")
                )
                
                if not preferences_changed:
                    logger.info(f"Using cached roadmap for user {user_id} (preferences unchanged)")
                    return cached_roadmap.get("roadmap_data")
                
                logger.info(f"Regenerating roadmap for user {user_id} (preferences changed)")
        
        # Generate new roadmap
        roadmap_data = generate_hybrid_roadmap(user_id)
        
        # Cache the new roadmap
        now = datetime.now()
        travel_prefs = get_user_travel_preferences(user_id)
        
        roadmaps_collection.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "user_id": user_id,
                    "roadmap_data": roadmap_data,
                    "travel_preferences": travel_prefs,
                    "created_at": now
                }
            },
            upsert=True
        )
        
        return roadmap_data
        
    except Exception as e:
        logger.error(f"Error getting roadmap with caching: {str(e)}")
        # Fallback to generating a new roadmap without caching
        return generate_hybrid_roadmap(user_id)
# --- PART 6: API ENDPOINTS (RECOMMENDATIONS) ---

@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {
        "success": True,
        "message": "Travel API v2.0.0 is running",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connectivity
        db_status = "connected"
        try:
            # Ping the database
            client.admin.command('ping')
        except Exception as e:
            db_status = f"disconnected: {str(e)}"
            
        # Check NLP model status
        nlp_status = "loaded"
        nlp_type = getattr(nlp, 'name', type(nlp).__name__)
        
        # Test if vectors are working
        test_doc = nlp("travel")
        has_vectors = hasattr(test_doc, 'vector_norm') and test_doc.vector_norm > 0
        
        if not has_vectors:
            nlp_status = "loaded without vectors"
            
        return {
            "success": True,
            "status": "healthy",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "database": db_status,
            "nlp_model": {
                "status": nlp_status,
                "type": nlp_type,
                "has_vectors": has_vectors
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "status": "unhealthy",
                "error": str(e)
            }
        )
@app.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: str,
    num: int = Query(10, ge=1, le=50),
    force_refresh: bool = Query(False),
    translate_results: bool = Query(False, description="Whether to translate results to user's preferred language"),
    language: Optional[str] = Query(None, description="Override language for translation"),
    background_tasks: BackgroundTasks = None
):
    """
    Get recommendations for a user with progressive pagination and translation support.
    """
    try:
        # Get recommendations with the enhanced progressive pagination
        recommendation_data = get_recommendations_with_caching(
            user_id, 
            force_refresh=force_refresh, 
            num_new_recommendations=num
        )
        
        # Combine new and previously shown recommendations
        all_recommendations = recommendation_data["new_recommendations"] + recommendation_data["previously_shown"]
        
        # Check if we need to regenerate cache - now using task_manager
        if background_tasks:
            cache_count = recommendations_cache_collection.count_documents({"user_id": user_id})
            if cache_count < 4:
                # Schedule cache regeneration with LOW priority
                logger.info(f"Cache count low ({cache_count}), scheduling regeneration")
                task_manager.schedule_task(
                    background_tasks,
                    TaskPriority.LOW,
                    background_cache_recommendations,
                    user_id, 
                    6
                )
        
        # Handle translation if requested
        target_language = language
        
        if translate_results and len(all_recommendations) > 0:
            # If language not specified, try to determine from user's search history
            if not target_language:
                # Get user's most recent search to determine language
                recent_search = search_queries_collection.find_one(
                    {"user_id": user_id},
                    sort=[("timestamp", -1)]
                )
                
                if recent_search and "detected_language" in recent_search:
                    target_language = recent_search["detected_language"]
            
            # Only translate if target language is valid and not English
            if target_language and target_language not in ['en', 'und']:
                translated_recommendations = []
                
                import copy
                
                # Dictionary of landmark types in different languages
                landmark_types = {
                    # Added English landmark types
                    "en": {
                        "Castle": "castle",
                        "Palace": "palace",
                        "Cathedral": "cathedral",
                        "Museum": "museum",
                        "Basilica": "basilica",
                        "Church": "church",
                        "Tower": "tower",
                        "Bridge": "bridge",
                        "Fort": "fort",
                        "Abbey": "abbey",
                        "Temple": "temple",
                        "Monument": "monument",
                        "Mosque": "mosque",
                        "Shrine": "shrine",
                        "Park": "park",
                        "Square": "square",
                        "Statue": "statue"
                    },
                    "fr": {
                        "Château": "castle/palace",
                        "Palais": "palace",
                        "Cathédrale": "cathedral",
                        "Musée": "museum",
                        "Basilique": "basilica",
                        "Église": "church",
                        "Tour": "tower",
                        "Pont": "bridge"
                    },
                    "es": {
                        "Palacio": "palace",
                        "Catedral": "cathedral",
                        "Museo": "museum",
                        "Iglesia": "church",
                        "Puente": "bridge"
                    },
                    "it": {
                        "Palazzo": "palace",
                        "Cattedrale": "cathedral",
                        "Museo": "museum",
                        "Chiesa": "church",
                        "Ponte": "bridge"
                    },
                    "de": {
                        "Schloss": "castle",
                        "Palast": "palace",
                        "Dom": "cathedral",
                        "Museum": "museum",
                        "Kirche": "church",
                        "Brücke": "bridge"
                    }
                }
                
                # Dictionary of known French chateau names and their Arabic transliterations
                known_transliterations = {
                    "ar": {
                        "Vaux-le-Vicomte": "فو لو فيكونت",
                        "Fontainebleau": "فونتينبلو",
                        "Haut-Koenigsbourg": "هاوت-كونيغسبورغ",
                        "Haut-kœnigsbourg": "هاوت-كونيغسبورغ",
                        "Chenonceau": "شينونسو",
                        "Chinonceau": "شينونسو",
                        "Saumur": "سومور",
                        "Tarascon": "تاراسكون",
                        "Versailles": "فرساي",
                        "Amboise": "أمبواز",
                        "Blois": "بلوا",
                        "Chambord": "شامبور",
                        "Cheverny": "شيفيرني",
                        "Carcassonne": "كاركاسون",
                        "Longchamp": "لونجشامب",
                        "Sully-sur-Loire": "سولي-سور-لوار",
                        "Beauregard": "بوريجارد",
                        "Chantilly": "شانتيي"
                    }
                }
                
                for place in all_recommendations:
                    # Deep copy to avoid modifying the original
                    translated_place = copy.deepcopy(place)
                    
                    # Improved place name translation
                    if "name" in translated_place and isinstance(translated_place["name"], str):
                        name = translated_place["name"]
                        name_language = detect_language(name)
                        
                        # If already in target language, skip translation
                        if name_language == target_language:
                            logger.info(f"Name already in target language ({name_language}): {name}")
                        else:
                            # Check if it's a landmark name in a known language
                            is_landmark = False
                            english_landmark_type = None
                            proper_name = None
                            
                            # Identify if this is a landmark and get its type
                            if name_language in landmark_types:
                                for landmark in landmark_types[name_language]:
                                    # Check for exact landmark or landmark followed by space
                                    if name == landmark or name.startswith(landmark + " "):
                                        is_landmark = True
                                        english_landmark_type = landmark_types[name_language][landmark]
                                        
                                        # Extract the proper name portion
                                        proper_name = name[len(landmark):].strip()
                                        
                                        # Handle common prepositions
                                        if name_language == "fr" and proper_name.startswith("de "):
                                            proper_name = proper_name[3:].strip()
                                        elif name_language == "fr" and proper_name.startswith("du "):
                                            proper_name = proper_name[3:].strip()
                                        elif name_language == "es" and proper_name.startswith("de "):
                                            proper_name = proper_name[3:].strip()
                                        elif name_language == "it" and proper_name.startswith("di "):
                                            proper_name = proper_name[3:].strip()
                                        elif name_language == "de" and proper_name.startswith("von "):
                                            proper_name = proper_name[4:].strip()
                                        elif name_language == "en" and proper_name.startswith("of "):
                                            proper_name = proper_name[3:].strip()
                                        
                                        break
                            
                            # Translation strategy based on name type
                            if is_landmark:
                                # First try direct translation of full name
                                try:
                                    from deep_translator import GoogleTranslator
                                    direct_translated = GoogleTranslator(source=name_language, target=target_language).translate(name)
                                    
                                    # If direct translation succeeded and changed the text
                                    if direct_translated != name and not any(char in direct_translated for char in name if ord(char) < 128):
                                        # Ensure the result doesn't contain any of the original Latin characters
                                        translated_place["name"] = direct_translated
                                        logger.info(f"Directly translated landmark name: {name} -> {translated_place['name']}")
                                    else:
                                        # Strategy 2: Translate only the landmark type
                                        translated_type = translate_from_english(english_landmark_type, target_language)
                                        
                                        # Handle proper name translation or transliteration
                                        if proper_name:
                                            if target_language == "ar":
                                                # Check if we have a known transliteration for this name
                                                clean_proper_name = proper_name.strip()
                                                if clean_proper_name in known_transliterations.get("ar", {}):
                                                    translated_proper_name = known_transliterations["ar"][clean_proper_name]
                                                    logger.info(f"Used known transliteration: {clean_proper_name} -> {translated_proper_name}")
                                                else:
                                                    # Try to use translator for transliteration
                                                    try:
                                                        transliterated_name = GoogleTranslator(source=name_language, target="ar").translate(clean_proper_name)
                                                        
                                                        # Check if we got a valid Arabic transliteration (no Latin characters)
                                                        if transliterated_name and not any(char in transliterated_name for char in clean_proper_name if ord(char) < 128):
                                                            translated_proper_name = transliterated_name
                                                            logger.info(f"Successfully transliterated: {clean_proper_name} -> {translated_proper_name}")
                                                        else:
                                                            # If transliteration has Latin chars, use original
                                                            translated_proper_name = clean_proper_name
                                                            logger.info(f"Transliteration failed, using original: {clean_proper_name}")
                                                    except Exception as e:
                                                        logger.warning(f"Error in transliteration: {e}, using original")
                                                        translated_proper_name = clean_proper_name
                                            else:
                                                # For non-Arabic targets, keep original proper names
                                                translated_proper_name = proper_name
                                        else:
                                            translated_proper_name = None
                                        
                                        # Combine translated parts
                                        if translated_proper_name:
                                            translated_place["name"] = f"{translated_type} {translated_proper_name}"
                                        else:
                                            translated_place["name"] = translated_type
                                        
                                        logger.info(f"Translated landmark name by parts: {name} -> {translated_place['name']}")
                                except Exception as e:
                                    logger.warning(f"Direct landmark translation failed: {e}, falling back to standard method")
                                    # Fall back to standard method
                                    if name_language == "en":
                                        translated_place["name"] = translate_from_english(name, target_language)
                                    else:
                                        english_name = translate_to_english(name)
                                        translated_place["name"] = translate_from_english(english_name, target_language)
                            else:
                                # Standard translation path for regular place names
                                if name_language == "en" or name_language == "und":
                                    translated_place["name"] = translate_from_english(name, target_language)
                                    logger.info(f"Translated name from English: {name} -> {translated_place['name']}")
                                else:
                                    # Try direct translation first for known languages
                                    try_direct = False
                                    
                                    # Some language pairs work better with direct translation
                                    if name_language in ["fr", "es", "de", "it", "ja", "zh", "ru", "ar"]:
                                        try_direct = True
                                    
                                    if try_direct:
                                        try:
                                            from deep_translator import GoogleTranslator
                                            direct_translated = GoogleTranslator(source=name_language, target=target_language).translate(name)
                                            if direct_translated != name:
                                                translated_place["name"] = direct_translated
                                                logger.info(f"Directly translated name: {name} -> {translated_place['name']}")
                                            else:
                                                # Fall back to English as intermediary
                                                english_name = translate_to_english(name)
                                                translated_place["name"] = translate_from_english(english_name, target_language)
                                                logger.info(f"Translated name via English: {name} ({name_language}) -> {english_name} -> {translated_place['name']}")
                                        except Exception:
                                            # Fall back to English as intermediary
                                            english_name = translate_to_english(name)
                                            translated_place["name"] = translate_from_english(english_name, target_language)
                                            logger.info(f"Translated name via English: {name} ({name_language}) -> {english_name} -> {translated_place['name']}")
                                    else:
                                        # Standard path via English
                                        english_name = translate_to_english(name)
                                        translated_place["name"] = translate_from_english(english_name, target_language)
                                        logger.info(f"Translated name via English: {name} ({name_language}) -> {english_name} -> {translated_place['name']}")
                    
                    # Description translation with type checking
                    if "description" in translated_place and isinstance(translated_place["description"], str):
                        description = translated_place["description"]
                        description_language = detect_language(description)
                        
                        if description_language == "en" or description_language == "und":
                            translated_place["description"] = translate_from_english(description, target_language)
                        elif description_language != target_language:
                            english_description = translate_to_english(description)
                            translated_place["description"] = translate_from_english(english_description, target_language)
                    
                    # Category translation with type checking
                    if "category" in translated_place and isinstance(translated_place["category"], str):
                        category = translated_place["category"]
                        category_language = detect_language(category)
                        
                        if category_language == "en" or category_language == "und":
                            translated_place["category"] = translate_from_english(category, target_language)
                        elif category_language != target_language:
                            english_category = translate_to_english(category)
                            translated_place["category"] = translate_from_english(english_category, target_language)
                    
                    # Tag translation with type checking
                    if "tags" in translated_place and isinstance(translated_place["tags"], list):
                        translated_tags = []
                        for tag in translated_place["tags"]:
                            if isinstance(tag, str):
                                tag_language = detect_language(tag)
                                
                                if tag_language == "en" or tag_language == "und":
                                    translated_tag = translate_from_english(tag, target_language)
                                elif tag_language != target_language:
                                    english_tag = translate_to_english(tag)
                                    translated_tag = translate_from_english(english_tag, target_language)
                                else:
                                    translated_tag = tag
                                    
                                translated_tags.append(translated_tag)
                            else:
                                translated_tags.append(tag)
                        translated_place["tags"] = translated_tags
                    
                    # Location fields translation
                    if "location" in translated_place and isinstance(translated_place["location"], dict):
                        if "city" in translated_place["location"] and isinstance(translated_place["location"]["city"], str):
                            city = translated_place["location"]["city"]
                            city_language = detect_language(city)
                            
                            if city_language == "en" or city_language == "und":
                                translated_place["location"]["city"] = translate_from_english(city, target_language)
                            elif city_language != target_language:
                                english_city = translate_to_english(city)
                                translated_place["location"]["city"] = translate_from_english(english_city, target_language)
                        
                        if "country" in translated_place["location"] and isinstance(translated_place["location"]["country"], str):
                            country = translated_place["location"]["country"]
                            country_language = detect_language(country)
                            
                            if country_language == "en" or country_language == "und":
                                translated_place["location"]["country"] = translate_from_english(country, target_language)
                            elif country_language != target_language:
                                english_country = translate_to_english(country)
                                translated_place["location"]["country"] = translate_from_english(english_country, target_language)
                    
                    translated_recommendations.append(translated_place)
                    
                all_recommendations = translated_recommendations
                logger.info(f"Translated {len(all_recommendations)} recommendations to {target_language}")
        
        # Simplify place objects in recommendations
        simplified_recommendations = []
        for place in all_recommendations:
            # Extract and clean average_rating
            average_rating = place.get("average_rating")
            # Handle MongoDB formatted number
            if isinstance(average_rating, dict) and "$numberDouble" in average_rating:
                try:
                    average_rating = float(average_rating.get("$numberDouble"))
                except (ValueError, TypeError):
                    average_rating = None
                
            simplified_place = {
                "place_id": place.get("_id"),
                "name": place.get("name"),
                "category": place.get("category"),
                "tags": place.get("tags", []),
                "description": place.get("description"),
                "location": {
                    "city": place.get("location", {}).get("city"),
                    "country": place.get("location", {}).get("country"),
                    "latitude": place.get("location", {}).get("latitude"),  # <- Added longitude
                    "longitude": place.get("location", {}).get("longitude")
                },
                "accessibility": place.get("accessibility", []),
                "average_rating": average_rating,  # Use the cleaned value
                "likes": place.get("likes"),
                "reviews_count": place.get("reviews_count"),
                "appropriate_time": place.get("appropriate_time", []),
                "budget": place.get("budget"),
                "group_type": place.get("group_type"),
                "image": place.get("image")
            }
            simplified_recommendations.append(simplified_place)
        
        # Create a structured response similar to roadmap format
        structured_response = {
            "data": []
        }
        
        # Add each place to the data array, without any message
        for place in simplified_recommendations:
            structured_response["data"].append({
                "place": place
            })
        
        # Return the structured response
        return structured_response
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
@app.post("/cache/generate/{user_id}")
async def force_cache_generation(
    user_id: str, 
    background_tasks: BackgroundTasks,
    num_entries: int = Query(6, ge=1, le=20),
    priority: str = Query(TaskPriority.MEDIUM, description="Task priority level")
):
    """
    Force cache generation for a user
    
    This is an admin endpoint to trigger cache generation
    """
    # Validate priority
    if priority not in [TaskPriority.HIGH, TaskPriority.MEDIUM, TaskPriority.LOW]:
        priority = TaskPriority.MEDIUM
        
    # Check if generation is already in progress
    cache_lock_key = f"cache_lock_{user_id}"
    lock = cache_locks_collection.find_one({"_id": cache_lock_key})
    
    if lock:
        # Check if the lock is stale (older than 5 minutes)
        lock_time = lock.get("timestamp", datetime.min)
        if isinstance(lock_time, str):
            try:
                lock_time = datetime.fromisoformat(lock_time.replace('Z', '+00:00'))
            except Exception:
                lock_time = datetime.min
                
        if (datetime.now() - lock_time).total_seconds() < 300:  # 5 minutes
            return {
                "success": True,
                "message": f"Cache generation already in progress for user {user_id}"
            }
    
    # Schedule cache generation with specified priority
    task_manager.schedule_task(
        background_tasks,
        priority,
        background_cache_recommendations,
        user_id,
        num_entries
    )
    
    return {
        "success": True,
        "message": f"Started {priority} priority generation of {num_entries} cache entries for user {user_id}"
    }

@app.get("/cache/status/{user_id}")
async def get_cache_status(user_id: str):
    """
    Get the status of the recommendation cache for a user
    
    Args:
        user_id: User ID to check cache for
    """
    try:
        # Check if lock exists
        cache_lock_key = f"cache_lock_{user_id}"
        lock = cache_locks_collection.find_one({"_id": cache_lock_key})
        
        # Get cached entries
        cached_entries = get_user_cached_recommendations(user_id)
        
        # Format timestamps
        if cached_entries:
            for entry in cached_entries:
                if "timestamp" in entry and not isinstance(entry["timestamp"], str):
                    entry["timestamp"] = entry["timestamp"].isoformat()
        
        return {
            "success": True,
            "user_id": user_id,
            "generation_in_progress": lock is not None,
            "cache_count": len(cached_entries),
            "cache_entries": [
                {
                    "sequence": entry.get("sequence"),
                    "timestamp": entry.get("timestamp")
                }
                for entry in cached_entries
            ]
        }
    except Exception as e:
        logger.error(f"Error getting cache status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
@app.get("/test_translation")
async def test_translation(
    text: str = "مرحبا بالعالم", 
    language: str = None, 
    translate_results: bool = False
):
    """
    Test the translation functionality
    
    Args:
        text: Text to translate
        language: Target language code (if translating from English)
        translate_results: Whether to translate to the target language
    """
    try:
        # First detect the source language
        detected = detect_language(text)
        
        # Initialize translation variable
        translated = text
        
        # If text is detected as English and we want to translate to another language
        if detected == "en" and language and language != "en" and translate_results:
            translated = translate_from_english(text, language)
            return {
                "original": text,
                "detected_language": detected,
                "target_language": language,
                "translated": translated,
                "success": True
            }
        # If text is not English, translate to English
        elif detected != "en":
            translated = translate_to_english(text)
            return {
                "original": text,
                "detected_language": detected,
                "translated": translated,
                "success": True
            }
        # If text is English but no target language or translation not requested
        else:
            return {
                "original": text,
                "detected_language": detected,
                "translated": text,  # No translation needed
                "success": True
            }
    except Exception as e:
        logger.error(f"Translation test failed: {e}")
        return {
            "original": text,
            "error": str(e),
            "success": False
        }
@app.delete("/cache/{user_id}")
async def clear_cache(user_id: str):
    """
    Clear the recommendation cache for a user
    
    Args:
        user_id: User ID to clear cache for
    """
    try:
        # Delete all cache entries
        result = recommendations_cache_collection.delete_many({"user_id": user_id})
        deleted_count = result.deleted_count
        
        # Also clear locks
        lock_result = cache_locks_collection.delete_one({"_id": f"cache_lock_{user_id}"})
        
        return {
            "success": True,
            "user_id": user_id,
            "deleted_count": deleted_count,
            "lock_cleared": lock_result.deleted_count > 0,
            "message": f"Cleared {deleted_count} cache entries for user {user_id}"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.delete("/shown-places/{user_id}")
async def reset_shown_places(user_id: str):
    """Reset all shown places for a user"""
    try:
        result = shown_places_collection.delete_many({"user_id": user_id})
        deleted_count = result.deleted_count
        
        return {
            "success": True,
            "user_id": user_id,
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} shown place records for user {user_id}"
        }
    except Exception as e:
        logger.error(f"Error resetting shown places: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/shown-places/{user_id}")
async def get_user_shown_places(user_id: str):
    """
    Get places that have been shown to a user
    
    Args:
        user_id: User ID
    """
    try:
        # Get previously shown places
        shown_place_ids = get_previously_shown_places(user_id)
        last_shown_place_ids = get_last_shown_places(user_id)
        
        # Get the actual place documents for the last shown
        last_shown_places = []
        if last_shown_place_ids:
            last_shown_places = list(places_collection.find({"_id": {"$in": last_shown_place_ids}}))
        
        return {
            "success": True,
            "user_id": user_id,
            "all_shown_count": len(shown_place_ids),
            "all_shown_ids": shown_place_ids,
            "last_shown_count": len(last_shown_place_ids),
            "last_shown_ids": last_shown_place_ids,
            "last_shown_places": last_shown_places
        }
    except Exception as e:
        logger.error(f"Error getting shown places: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
# --- PART 7: API ENDPOINTS (SEARCH AND ROADMAP) ---

@app.get("/search/{user_id}")
async def search_places(
    user_id: str,
    query: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    translate_results: bool = Query(False, description="Whether to translate results back to the query language"),
    language: Optional[str] = Query(None, description="Override detected language")
):
    """
    Search for places based on a text query with language support
    
    Args:
        user_id: User ID (for tracking)
        query: Search query string
        limit: Maximum number of results to return
        translate_results: Whether to translate results back to the query language
        language: Override automatically detected language
    """
    try:
        # Store original query
        original_query = query
        
        # Use provided language or detect language
        detected_language = language if language else detect_language(query)
        
        # Translate query to English if needed
        if detected_language != 'en':
            # Translate query to English for better matching
            translated_query = translate_to_english(query)
            logger.info(f"Translated search query from '{original_query}' ({detected_language}) to '{translated_query}'")
            query = translated_query
        
        # Track search query with translation info
        search_queries_collection.insert_one({
            "user_id": user_id,
            "query": original_query,
            "translated_query": query if detected_language != 'en' else None,
            "detected_language": detected_language,
            "timestamp": datetime.now()
        })
        
        # Get all places
        all_places = list(places_collection.find())
        
        # Improved method to score results with semantic search
        results = []
        
        # Check if NLP model has word vectors
        test_doc = nlp("test")
        has_vectors = hasattr(test_doc, 'vector_norm') and test_doc.vector_norm > 0
        
        if has_vectors:
            # Use semantic search for better matching
            query_doc = nlp(query.lower())
            
            for place in all_places:
                # Initialize score components
                name_score = 0
                desc_score = 0
                tag_score = 0
                category_score = 0
                
                # 1. Exact name match (highest weight)
                if query.lower() in place.get("name", "").lower():
                    name_score = 0.9  # Direct substring match
                
                # Try semantic match on name
                place_name = place.get("name", "")
                if place_name:
                    place_name_doc = nlp(place_name.lower())
                    name_similarity = query_doc.similarity(place_name_doc)
                    name_score = max(name_score, name_similarity * 0.8)  # Max of exact or semantic
                
                # 2. Description match
                description = place.get("description", "")
                if description:
                    if query.lower() in description.lower():
                        desc_score = 0.5  # Direct substring match in description
                    
                    # Semantic similarity for description (if not too long)
                    if len(description) < 1000:  # Avoid processing very long descriptions
                        desc_doc = nlp(description.lower())
                        desc_score = max(desc_score, query_doc.similarity(desc_doc) * 0.5)
                
                # 3. Tags match
                tags = place.get("tags", [])
                if tags:
                    # Check for direct tag matches
                    if query.lower() in [tag.lower() for tag in tags]:
                        tag_score = 0.8  # Direct tag match
                    
                    # Semantic similarity for tags
                    max_tag_similarity = 0
                    for tag in tags:
                        tag_doc = nlp(tag.lower())
                        similarity = query_doc.similarity(tag_doc)
                        max_tag_similarity = max(max_tag_similarity, similarity)
                    
                    tag_score = max(tag_score, max_tag_similarity * 0.7)
                
                # 4. Category match
                category = place.get("category", "")
                if category:
                    if query.lower() in category.lower():
                        category_score = 0.7  # Direct category match
                    
                    # Semantic similarity for category
                    category_doc = nlp(category.lower())
                    category_score = max(category_score, query_doc.similarity(category_doc) * 0.6)
                
                # Compute final score with weights
                # Name (40%), Tags (30%), Category (20%), Description (10%)
                final_score = (
                    0.4 * name_score +
                    0.3 * tag_score +
                    0.2 * category_score +
                    0.1 * desc_score
                )
                
                # Only add results with a minimum relevance
                if final_score > 0.3:  # Threshold for relevance
                    results.append({
                        "place": place,
                        "score": final_score
                    })
                
        else:
            # Fallback to basic text matching if vectors aren't available
            logger.warning("Word vectors not available, using basic text search")
            
            for place in all_places:
                score = 0
                
                # Exact name match - highest score
                if query.lower() in place.get("name", "").lower():
                    score = 1.0
                # Tag match - high score
                elif "tags" in place and any(query.lower() in tag.lower() for tag in place.get("tags", [])):
                    score = 0.8
                # Category match - medium score
                elif "category" in place and query.lower() in place.get("category", "").lower():
                    score = 0.7
                # Description match - lower score
                elif "description" in place and query.lower() in place.get("description", "").lower():
                    score = 0.5
                    
                if score > 0:
                    results.append({
                        "place": place,
                        "score": score
                    })
                
        # Sort by score (highest first) and limit results
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)[:limit]
        
        # Extract just the place data
        places_only = [item["place"] for item in sorted_results]
        
        # Translate results back to original language if requested
        if translate_results and detected_language not in ['en', 'und'] and len(places_only) > 0:
            translated_places = []
            
            import copy
            
            for place in places_only:
                # Deep copy to avoid modifying the original
                translated_place = copy.deepcopy(place)
                
                # Improve name translation with language detection
                if "name" in translated_place and isinstance(translated_place["name"], str):
                    name = translated_place["name"]
                    
                    # Check if name is already in the target language or in a non-English language
                    name_language = detect_language(name)
                    
                    if name_language == "en" or name_language == "und":
                        # If name is English or undetermined, translate from English
                        translated_place["name"] = translate_from_english(name, detected_language)
                        logger.info(f"Translated name from English: {name} -> {translated_place['name']}")
                    elif name_language != detected_language:
                        # If name is in a different language (not English, not target language)
                        # First translate to English, then to target language
                        english_name = translate_to_english(name)
                        translated_place["name"] = translate_from_english(english_name, detected_language)
                        logger.info(f"Translated name via English: {name} ({name_language}) -> {english_name} -> {translated_place['name']}")
                    # If name is already in target language, leave it as is
                    else:
                        logger.info(f"Name already in target language ({name_language}): {name}")
                
                # Description translation with type checking
                if "description" in translated_place and isinstance(translated_place["description"], str):
                    description = translated_place["description"]
                    description_language = detect_language(description)
                    
                    if description_language == "en" or description_language == "und":
                        translated_place["description"] = translate_from_english(description, detected_language)
                    elif description_language != detected_language:
                        english_description = translate_to_english(description)
                        translated_place["description"] = translate_from_english(english_description, detected_language)
                
                # Category translation with type checking
                if "category" in translated_place and isinstance(translated_place["category"], str):
                    category = translated_place["category"]
                    category_language = detect_language(category)
                    
                    if category_language == "en" or category_language == "und":
                        translated_place["category"] = translate_from_english(category, detected_language)
                    elif category_language != detected_language:
                        english_category = translate_to_english(category)
                        translated_place["category"] = translate_from_english(english_category, detected_language)
                
                # Tag translation with type checking
                if "tags" in translated_place and isinstance(translated_place["tags"], list):
                    translated_tags = []
                    for tag in translated_place["tags"]:
                        if isinstance(tag, str):
                            tag_language = detect_language(tag)
                            
                            if tag_language == "en" or tag_language == "und":
                                translated_tag = translate_from_english(tag, detected_language)
                            elif tag_language != detected_language:
                                english_tag = translate_to_english(tag)
                                translated_tag = translate_from_english(english_tag, detected_language)
                            else:
                                translated_tag = tag
                                
                            translated_tags.append(translated_tag)
                        else:
                            translated_tags.append(tag)
                    translated_place["tags"] = translated_tags
                
                # Location fields translation
                if "location" in translated_place and isinstance(translated_place["location"], dict):
                    if "city" in translated_place["location"] and isinstance(translated_place["location"]["city"], str):
                        city = translated_place["location"]["city"]
                        city_language = detect_language(city)
                        
                        if city_language == "en" or city_language == "und":
                            translated_place["location"]["city"] = translate_from_english(city, detected_language)
                        elif city_language != detected_language:
                            english_city = translate_to_english(city)
                            translated_place["location"]["city"] = translate_from_english(english_city, detected_language)
                    
                    if "country" in translated_place["location"] and isinstance(translated_place["location"]["country"], str):
                        country = translated_place["location"]["country"]
                        country_language = detect_language(country)
                        
                        if country_language == "en" or country_language == "und":
                            translated_place["location"]["country"] = translate_from_english(country, detected_language)
                        elif country_language != detected_language:
                            english_country = translate_to_english(country)
                            translated_place["location"]["country"] = translate_from_english(english_country, detected_language)
                
                translated_places.append(translated_place)
                
            places_only = translated_places
            logger.info(f"Translated {len(places_only)} results to {detected_language}")
        
        # Simplify the place objects in the final results
        simplified_results = []
        for place in places_only:
            simplified_place = {
                "place_id": place.get("_id"),
                "name": place.get("name"),
                "category": place.get("category"),
                "tags": place.get("tags", []),
                "description": place.get("description"),
                "location": {
                    "city": place.get("location", {}).get("city"),
                    "country": place.get("location", {}).get("country"),
                    "latitude": place.get("location", {}).get("latitude"),
                    "longitude": place.get("location", {}).get("longitude")
                },
                "accessibility": place.get("accessibility", []),
                "average_rating": place.get("average_rating"),
                "likes": place.get("likes"),
                "reviews_count": place.get("reviews_count"),
                "appropriate_time": place.get("appropriate_time", []),
                "budget": place.get("budget"),
                "group_type": place.get("group_type"),
                "image": place.get("image")
            }
            simplified_results.append(simplified_place)
        
        # Create entries with place objects
        place_entries = []
        for place in simplified_results:
            place_entries.append({
                "place": place
            })
        
        # Create a structured response
        structured_response = {
            "data": place_entries
        }
        
        # Add message outside the data array
        if not simplified_results:
            structured_response["message"] = {
                "type": "message",
                "message_type": "no_results",
                "message": f"No places found matching '{original_query}'. Try a different search term.",
                "is_warning": True
            }
        else:
            structured_response["message"] = {
                "type": "message",
                "message_type": "success",
                "message": f"Found {len(simplified_results)} places matching your search.",
                "is_warning": False
            }
        
        # Return the structured response
        return structured_response
        
    except Exception as e:
        logger.error(f"Error searching places: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
@app.get("/search-history/{user_id}")
async def get_search_history(
    user_id: str,
    limit: int = Query(10, ge=1, le=50)
):
    """
    Get search history for a user, including original and translated queries
    
    Args:
        user_id: User ID
        limit: Maximum number of results to return
    """
    try:
        # Get search history sorted by newest first
        # Include translated_query and detected_language fields
        history = list(
            search_queries_collection.find(
                {"user_id": user_id},
                {
                    "_id": 0, 
                    "user_id": 1, 
                    "query": 1, 
                    "translated_query": 1,
                    "detected_language": 1,
                    "timestamp": 1
                }
            )
            .sort("timestamp", -1)
            .limit(limit)
        )
        
        # Format timestamps and handle older records without translation fields
        for item in history:
            # Format timestamp
            if "timestamp" in item and not isinstance(item["timestamp"], str):
                item["timestamp"] = item["timestamp"].isoformat()
            
            # Ensure translation fields exist (for backward compatibility)
            if "detected_language" not in item:
                item["detected_language"] = "en"
            
            if "translated_query" not in item and "query" in item:
                # If this is an old record without translation, they're the same
                item["translated_query"] = None
        
        return {
            "success": True,
            "user_id": user_id,
            "count": len(history),
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Error getting search history: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
@app.delete("/search-history/{user_id}")
async def clear_search_history(user_id: str):
    """
    Clear search history for a user
    
    Args:
        user_id: User ID
    """
    try:
        result = search_queries_collection.delete_many({"user_id": user_id})
        deleted_count = result.deleted_count
        
        return {
            "success": True,
            "user_id": user_id,
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} search history records for user {user_id}"
        }
    except Exception as e:
        logger.error(f"Error clearing search history: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# --- Roadmap API Endpoints ---
@app.get("/roadmap/{user_id}")
async def get_roadmap(user_id: str, language: str = None):
    try:
        logger.info(f"Roadmap request for user {user_id}")
        
        # Get roadmap (cached or newly generated) - now using task_manager with HIGH priority
        roadmap_data = await task_manager.run_task(
            TaskPriority.HIGH,
            get_roadmap_with_caching, 
            user_id
        )
        
        # Simplify to list format
        simplified_list = simplify_roadmap_to_list(roadmap_data)
        
        # Translate if language parameter is provided
        if language:
            logger.info(f"Translating roadmap to {language}")
            simplified_list = translate_roadmap_results(simplified_list, language)
        
        return {
            "data": simplified_list
            # Entire metadata section removed
        }
    except Exception as e:
        logger.error(f"Error generating roadmap: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
@app.post("/roadmap")
async def create_roadmap(request: RoadmapRequest):
    """
    Generate a new travel roadmap for a user (force regeneration)
    """
    try:
        user_id = request.user_id
        logger.info(f"Force regenerating roadmap for user {user_id}")
        
        # Force generation of new roadmap
        roadmap_data = generate_hybrid_roadmap(user_id)
        
        # Store in cache
        now = datetime.now()
        travel_prefs = get_user_travel_preferences(user_id)
        roadmaps_collection.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "user_id": user_id,
                    "roadmap_data": roadmap_data,
                    "travel_preferences": travel_prefs,
                    "created_at": now
                }
            },
            upsert=True
        )
        
        # Simplify to list format
        simplified_list = simplify_roadmap_to_list(roadmap_data)
        
        return {
            "success": True, 
            "user_id": user_id, 
            "count": len(simplified_list),
            "data": simplified_list,
            "metadata": {
                "budget_level": roadmap_data.get("budget_level"),
                "group_type": roadmap_data.get("group_type"),
                "start_date": roadmap_data.get("start_date"),
                "accessibility_needs": roadmap_data.get("accessibility_needs", [])
            }
        }
    except Exception as e:
        logger.error(f"Error generating roadmap: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/roadmap-status/{user_id}")
async def get_roadmap_cache_status(user_id: str):
    """
    Get the status of the roadmap cache for a user
    
    Args:
        user_id: User ID
    """
    try:
        # Check if roadmap exists in cache
        cached_roadmap = roadmaps_collection.find_one({"user_id": user_id})
        
        if cached_roadmap:
            # Format created_at timestamp
            created_at = cached_roadmap.get("created_at")
            if created_at and not isinstance(created_at, str):
                created_at = created_at.isoformat()
                
            # Get current preferences
            current_prefs = get_user_travel_preferences(user_id)
            cached_prefs = cached_roadmap.get("travel_preferences")
            
            # Compare preferences to check if they've changed
            preferences_changed = True
            if current_prefs and cached_prefs:
                preferences_changed = (
                    current_prefs.get("budget") != cached_prefs.get("budget") or
                    current_prefs.get("accessibility_needs") != cached_prefs.get("accessibility_needs") or
                    current_prefs.get("group_type") != cached_prefs.get("group_type") or
                    current_prefs.get("travel_dates") != cached_prefs.get("travel_dates") or
                    current_prefs.get("destinations") != cached_prefs.get("destinations")
                )
            
            return {
                "success": True,
                "user_id": user_id,
                "cache_exists": True,
                "created_at": created_at,
                "place_count": len(cached_roadmap.get("roadmap_data", {}).get("places", [])),
                "preferences_changed": preferences_changed,
                "cached_preferences": cached_prefs,
                "current_preferences": current_prefs
            }
        else:
            return {
                "success": True,
                "user_id": user_id,
                "cache_exists": False
            }
    except Exception as e:
        logger.error(f"Error getting roadmap cache status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.delete("/roadmap/{user_id}")
async def clear_roadmap_cache(user_id: str):
    """
    Clear the roadmap cache for a specific user
    """
    try:
        result = roadmaps_collection.delete_one({"user_id": user_id})
        deleted = result.deleted_count > 0
        
        return {
            "success": True,
            "user_id": user_id,
            "cache_cleared": deleted
        }
    except Exception as e:
        logger.error(f"Error clearing roadmap cache: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
# --- PART 8: ERROR HANDLERS AND SERVER STARTUP ---

# --- Place and User Endpoints ---

@app.get("/place/{place_id}")
async def get_place(place_id: str):
    """
    Get details for a specific place
    
    Args:
        place_id: Place ID
    """
    try:
        place = places_collection.find_one({"_id": place_id})
        
        if not place:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": f"Place {place_id} not found"}
            )
            
        return {
            "success": True,
            "place": place
        }
    except Exception as e:
        logger.error(f"Error getting place details: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/places/stats")
async def get_places_stats():
    """Get statistics about places in the database"""
    try:
        total_count = places_collection.count_documents({})
        
        # Get category distribution
        category_pipeline = [
            {"$group": {"_id": "$category", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        categories = list(places_collection.aggregate(category_pipeline))
        
        # Get average rating
        rating_pipeline = [
            {"$match": {"average_rating": {"$exists": True}}},
            {"$group": {"_id": None, "avg": {"$avg": "$average_rating"}}}
        ]
        avg_rating_result = list(places_collection.aggregate(rating_pipeline))
        avg_rating = avg_rating_result[0]["avg"] if avg_rating_result else 0
        
        return {
            "success": True,
            "total_places": total_count,
            "categories": categories,
            "average_rating": avg_rating
        }
    except Exception as e:
        logger.error(f"Error getting places stats: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/user-preferences/{user_id}")
async def get_user_preferences_endpoint(user_id: str):
    """
    Get user preferences
    
    Args:
        user_id: User ID
    """
    try:
        # Get general preferences
        preferences = get_user_preferences(user_id)
        
        # Get travel preferences
        travel_prefs = get_user_travel_preferences(user_id)
        
        if not preferences and not travel_prefs:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": f"No preferences found for user {user_id}"}
            )
            
        return {
            "success": True,
            "user_id": user_id,
            "general_preferences": preferences,
            "travel_preferences": travel_prefs
        }
    except Exception as e:
        logger.error(f"Error getting user preferences: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/system/stats")
async def get_system_stats():
    """Get system statistics for monitoring"""
    try:
        # Get collection counts
        collection_stats = {}
        
        for collection_name in [
            "users", "places", "interactions", "search_queries", 
            "user_travel_preferences", "recommendations_cache", 
            "shown_places", "roadmaps", "cache_locks"
        ]:
            collection_stats[collection_name] = db[collection_name].count_documents({})
            
        # Get NLP model info
        nlp_info = {
            "name": getattr(nlp, "name", str(type(nlp).__name__)),
            "has_vectors": nlp.vocab.vectors.n_keys > 0
        }
        
        # Memory usage of recent cache entries (sample)
        cache_sample = list(recommendations_cache_collection.find().limit(1))
        cache_size = len(str(cache_sample)) if cache_sample else 0
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "collection_stats": collection_stats,
            "nlp_model": nlp_info,
            "cache_sample_size_bytes": cache_size,
            "api_version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/debug/nlp-test")
async def test_nlp():
    """Test NLP model with word similarity comparisons"""
    try:
        # Check if model has vectors
        test_word = "travel"
        test_doc = nlp(test_word)
        has_vectors = hasattr(test_doc, 'vector_norm') and test_doc.vector_norm > 0
        
        if not has_vectors:
            return {
                "success": False,
                "error": "NLP model does not have word vectors",
                "model": getattr(nlp, "name", str(type(nlp).__name__)),
                "fallback_active": isinstance(nlp, type) and nlp.__name__ == "DummyNLP"
            }
            
        # Test word similarity pairs
        word_pairs = [
            ("beach", "ocean"),
            ("mountain", "hiking"),
            ("museum", "history"),
            ("restaurant", "food"),
            ("hotel", "accommodation")
        ]
        
        similarity_results = {}
        for word1, word2 in word_pairs:
            doc1 = nlp(word1)
            doc2 = nlp(word2)
            similarity = doc1.similarity(doc2)
            similarity_results[f"{word1}_{word2}"] = similarity
            
        return {
            "success": True,
            "has_vectors": has_vectors,
            "model": getattr(nlp, "name", str(type(nlp).__name__)),
            "similarity_results": similarity_results,
            "vector_examples": {
                "travel": test_doc.vector[:5].tolist(),  # First 5 dimensions
                "norm": float(test_doc.vector_norm)
            }
        }
    except Exception as e:
        logger.error(f"Error testing NLP model: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# --- Error Handlers ---

@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"success": False, "error": "Resource not found"}
    )

@app.exception_handler(500)
async def server_exception_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"}
    )

# --- Server Startup ---
if __name__ == "__main__":
    import uvicorn
    
    # Check environment for port
    port = int(os.environ.get("PORT", 8000))
    
    # Log startup information
    logger.info("=" * 50)
    logger.info("Starting Travel API Server v2.0.0")
    logger.info(f"Using port: {port}")
    
    # Check MongoDB connection
    try:
        client.server_info()
        logger.info("✅ MongoDB connection verified")
        
        # Log collection counts
        for coll_name in ["users", "places", "recommendations_cache"]:
            count = db[coll_name].count_documents({})
            logger.info(f"Collection {coll_name}: {count} documents")
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        
    # Log NLP model status
    has_vectors = nlp.vocab.vectors.n_keys > 0
    logger.info(f"NLP Model: {getattr(nlp, 'name', type(nlp).__name__)}")
    logger.info(f"Word Vectors: {'Available' if has_vectors else 'NOT AVAILABLE'}")
    logger.info("=" * 50)
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=port)
