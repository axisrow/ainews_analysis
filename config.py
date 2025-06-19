#!/usr/bin/env python3
"""
AI News Research Project Configuration
Python configuration file to replace YAML
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# AI News Research Project Configuration
CONFIG = {
    # Data Collection Settings
    "data_collection": {
        # Search keywords for all sources
        "search_keywords": [
            "artificial intelligence",
            "machine learning", 
            "deep learning",
            "neural networks",
            "AI ethics",
            "GPT",
            "ChatGPT",
            "AI regulation",
            "computer vision",
            "natural language processing"
        ],
        
        # Web scraping settings
        "web_scraping": {
            "sites": [
                {
                    "name": "MIT Technology Review",
                    "url": "https://www.technologyreview.com/artificial-intelligence/",
                    "article_selector": "div.contentCard, article.teaserItem",
                    "title_selector": "h3 a, h2 a, .teaserItem__title a",
                    "content_selector": "p, .teaserItem__excerpt"
                },
                {
                    "name": "VentureBeat AI", 
                    "url": "https://venturebeat.com/ai/",
                    "article_selector": "article.ArticleListing, .ArticleCard",
                    "title_selector": "h2 a, h3 a, .ArticleCard__title a",
                    "content_selector": "p, .ArticleCard__excerpt"
                },
                {
                    "name": "The Verge AI",
                    "url": "https://www.theverge.com/ai-artificial-intelligence", 
                    "article_selector": "article, .c-entry-box",
                    "title_selector": "h2 a, h3 a, .c-entry-box--compact__title a",
                    "content_selector": "p, .c-entry-summary"
                }
            ],
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "headers": {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Cache-Control": "max-age=0"
            },
            "timeout": 30,
            "delay_between_requests": 1.0,  # Increased delay to avoid rate limiting
            "max_articles_per_site": 100,  # Максимум статей с одного сайта
            "max_retries": 3,  # Number of retries for failed requests
            "backoff_factor": 2  # Exponential backoff multiplier
        },
        
        # RSS feeds - Verified working sources 2024
        "rss_feeds": [
            # Core AI News Sources (tested working)
            {"url": "https://feeds.feedburner.com/venturebeat/SZYF", "name": "VentureBeat AI"},
            {"url": "https://techcrunch.com/category/artificial-intelligence/feed/", "name": "TechCrunch AI"}, 
            {"url": "https://www.technologyreview.com/feed/", "name": "MIT Tech Review"},
            {"url": "https://www.artificialintelligence-news.com/feed/", "name": "AI News"},
            {"url": "https://feeds.arstechnica.com/arstechnica/index", "name": "Ars Technica"},
            {"url": "https://www.theverge.com/rss/index.xml", "name": "The Verge"},
            {"url": "https://www.zdnet.com/topic/artificial-intelligence/rss.xml", "name": "ZDNet AI"},
            {"url": "https://venturebeat.com/feed/", "name": "VentureBeat All"},
            {"url": "https://singularityhub.com/feed/", "name": "Singularity Hub"},
            
            # Academic/Research Sources (verified working)
            {"url": "https://bair.berkeley.edu/blog/feed.xml", "name": "Berkeley AI Research"},
            {"url": "https://blog.ml.cmu.edu/feed", "name": "CMU ML Blog"},
            {"url": "https://www.marktechpost.com/feed", "name": "MarkTechPost"},
            {"url": "https://machinelearningmastery.com/feed/", "name": "ML Mastery"},
            # Removed Papers With Code - returns 404 error
            
            # Major AI Companies (selective working ones)
            {"url": "https://blog.google/technology/ai/rss/", "name": "Google AI Blog"},
            {"url": "https://huggingface.co/blog/feed.xml", "name": "Hugging Face Blog"},
            
            # Technical/Research Sources
            {"url": "https://blog.tensorflow.org/feeds/posts/default", "name": "TensorFlow Blog"},
            {"url": "https://pytorch.org/blog/feed.xml", "name": "PyTorch Blog"},
            {"url": "https://research.googleblog.com/feeds/posts/default", "name": "Google Research Blog"},
            {"url": "https://aws.amazon.com/blogs/machine-learning/feed/", "name": "AWS ML Blog"},
            {"url": "https://towardsdatascience.com/feed", "name": "Towards Data Science"},
            
            # Business/Industry Focus  
            # Removed Unite AI - has XML syntax errors
            {"url": "https://analyticsindiamag.com/feed/", "name": "Analytics India Magazine"}
        ],
        
        # Reddit scraping - DISABLED due to API authentication requirements since 2023
        "reddit": {
            "enabled": False,  # Reddit now requires OAuth authentication for all API access
            "note": "Reddit API changes in 2023 require OAuth authentication and paid access for commercial use",
            "subreddits": [],  # Disabled - keeping empty for future potential re-enablement
            "sort_by": "top",
            "time_filter": "year", 
            "limit": 0  # Disabled
        }
    },

    # Database Configuration
    "database": {
        "type": "sqlite",
        "path": "data/ai_news.db"
    },

    # NLP Analysis Settings
    "nlp": {
        "language": "en",
        "spacy_model": "en_core_web_sm",
        
        # Sentiment analysis
        "sentiment": {
            "model": "cardiffnlp/twitter-roberta-base-sentiment"
        },
        
        # Topic modeling
        "topics": {
            "num_topics": 10,
            "method": "lda"  # or "nmf"
        },
        
        # Named Entity Recognition
        "ner": {
            "enabled": True,
            "entities": [
                "ORG",      # Organizations
                "PERSON",   # People
                "PRODUCT",  # Products
                "DATE",     # Dates
                "MONEY"     # Financial amounts
            ]
        },
        
        # Google GenAI Analysis (New SDK)
        "genai": {
            "enabled": True,
            "model": "gemini-2.0-flash-001",  # Latest model with improved capabilities
            "api_key": os.getenv("GOOGLE_API_KEY", ""),  # Load from environment variable
            "max_retries": 3,
            "delay_between_requests": 1.0,  # seconds between requests
            "batch_size": 10,  # articles to process in parallel
            "temperature": 0.7,  # Creativity level (0.0-1.0)
            "use_async": False,  # Enable async processing for better performance
            "async_batch_size": 5,  # Concurrent requests in async mode
            "safety_settings": {
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",  # Safety threshold
                "enable_content_filtering": True
            },
            "structured_output": True,  # Use structured JSON responses
            "fallback_on_error": True  # Use fallback analysis if API fails
        }
    },

    # Data Processing
    "processing": {
        "batch_size": 100,
        "max_workers": 4,
        
        # Text preprocessing
        "preprocessing": {
            "remove_stopwords": True,
            "lemmatization": True,
            "min_word_length": 3,
            "max_word_length": 30
        }
    },

    # Visualization Settings
    "visualization": {
        "dashboard": {
            "port": 8501,
            "theme": "dark"
        },
        
        "charts": {
            "color_scheme": "viridis",
            "figure_size": [10, 6]
        },
        
        "wordcloud": {
            "max_words": 100,
            "background_color": "white",
            "colormap": "viridis"
        }
    },

    # Scheduling
    "schedule": {
        "enabled": False,
        "interval_hours": 6  # Collect data every 6 hours
    },

    # Logging
    "logging": {
        "level": "INFO",
        "format": "{time} | {level} | {message}",
        "file": "logs/ai_news_research.log",
        "rotation": "1 week",
        "retention": "1 month"
    },

    # Output Settings
    "output": {
        "reports_dir": "reports",
        "data_dir": "data",
        
        # Export formats
        "export_formats": [
            "csv",
            "json", 
            "excel"
        ]
    },

    # Test Mode Settings
    "test_mode": {
        "max_articles_to_analyze": 5,
        "max_articles_per_source": 10,
        "reduced_delays": True,
        "verbose_logging": False
    }
}


def get_config():
    """Get the configuration dictionary"""
    return CONFIG


def get_google_api_key():
    """Get Google API key from environment"""
    return os.getenv("GOOGLE_API_KEY", "")


def set_test_mode():
    """Configure for test mode with reduced limits"""
    test_settings = CONFIG["test_mode"]
    
    # Apply test mode settings
    CONFIG["data_collection"]["reddit"]["limit"] = test_settings["max_articles_per_source"]
    CONFIG["data_collection"]["web_scraping"]["max_articles_per_site"] = test_settings["max_articles_per_source"]
    
    if test_settings["reduced_delays"]:
        CONFIG["data_collection"]["web_scraping"]["delay_between_requests"] = 1
        CONFIG["nlp"]["genai"]["delay_between_requests"] = 0.5
    
    if test_settings["verbose_logging"]:
        CONFIG["logging"]["level"] = "DEBUG"
    
    return CONFIG


def set_bulk_mode():
    """Configure for bulk mode with maximum data collection"""
    CONFIG["data_collection"]["reddit"]["limit"] = 1000
    CONFIG["data_collection"]["web_scraping"]["delay_between_requests"] = 0.5
    CONFIG["data_collection"]["web_scraping"]["max_articles_per_site"] = 200
    return CONFIG


def update_genai_settings(api_key=None, enabled=None, model=None):
    """Update GenAI settings"""
    if api_key is not None:
        CONFIG["nlp"]["genai"]["api_key"] = api_key
    if enabled is not None:
        CONFIG["nlp"]["genai"]["enabled"] = enabled
    if model is not None:
        CONFIG["nlp"]["genai"]["model"] = model
    return CONFIG


# For backward compatibility
def load_config():
    """Load configuration (backward compatibility function)"""
    return get_config()