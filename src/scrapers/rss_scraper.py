import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
import hashlib
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class RSSFeedScraper:
    def __init__(self, config: Dict):
        self.config = config
        self.keywords = [kw.lower() for kw in config.get('search_keywords', [])]
        rss_settings = config.get('rss_settings', {})
        self.max_age_days = rss_settings.get('max_article_age_days', 7)
        self.min_date = datetime.now() - timedelta(days=self.max_age_days)
        
    def parse_feed(self, feed_config: Dict) -> List[Dict]:
        """Parse a single RSS feed and extract AI-related articles with validation"""
        articles = []
        
        try:
            logger.info(f"Parsing RSS feed: {feed_config['name']}")
            
            # Validate feed first
            if not self._validate_feed(feed_config['url']):
                logger.debug(f"Skipping invalid RSS feed: {feed_config['name']}")
                return articles
            
            feed = feedparser.parse(feed_config['url'])
            
            # Check for parsing errors
            if feed.bozo:
                # Only log as warning for minor issues, skip for major ones
                if isinstance(feed.bozo_exception, (feedparser.CharacterEncodingOverride, 
                                                   feedparser.NonXMLContentType)):
                    logger.debug(f"Feed parsing warning for {feed_config['name']}: {feed.bozo_exception}")
                else:
                    logger.warning(f"Feed parsing error for {feed_config['name']}: {type(feed.bozo_exception).__name__}")
                    return articles
            
            # Check if feed has entries
            if not hasattr(feed, 'entries') or len(feed.entries) == 0:
                logger.warning(f"No entries found in RSS feed: {feed_config['name']}")
                return articles
            
            for entry in feed.entries:
                article = self._extract_article_from_entry(entry, feed_config['name'])
                
                if article and self._is_ai_related(article):
                    articles.append(article)
                    
        except Exception as e:
            logger.error(f"Error parsing RSS feed {feed_config['name']}: {str(e)}")
            
        return articles
    
    def _extract_article_from_entry(self, entry: Any, source_name: str) -> Optional[Dict]:
        """Extract article information from RSS entry"""
        try:
            article = {
                'source': source_name,
                'source_type': 'rss',
                'scraped_at': datetime.now()
            }
            
            # Extract title
            if hasattr(entry, 'title'):
                article['title'] = entry.title
            
            # Extract URL
            if hasattr(entry, 'link'):
                article['url'] = entry.link
                article['id'] = hashlib.md5(entry.link.encode()).hexdigest()
            
            # Extract summary/description
            if hasattr(entry, 'summary'):
                article['summary'] = entry.summary
            elif hasattr(entry, 'description'):
                article['summary'] = entry.description
            
            # Extract publish date
            published_date = None
            if hasattr(entry, 'published_parsed'):
                published_date = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'updated_parsed'):
                published_date = datetime(*entry.updated_parsed[:6])
            
            if published_date:
                # Check if article is too old
                if published_date < self.min_date:
                    logger.debug(f"Article too old ({published_date}): {article.get('title', 'Unknown')[:50]}")
                    return None  # Skip old articles
                
                article['published_date'] = published_date
                article['date_extraction_attempted'] = True
            else:
                # No date found in RSS entry
                article['date_extraction_attempted'] = True
                article['needs_manual_review'] = True
            
            # Extract author
            if hasattr(entry, 'author'):
                article['author'] = entry.author
            
            # Extract tags/categories
            tags = []
            if hasattr(entry, 'tags'):
                tags.extend([tag.term for tag in entry.tags])
            if hasattr(entry, 'categories'):
                tags.extend([cat for cat in entry.categories])
            if tags:
                article['tags'] = list(set(tags))
            
            return article if 'title' in article and 'url' in article else None
            
        except Exception as e:
            logger.warning(f"Error extracting article from RSS entry: {str(e)}")
            return None
    
    def _is_ai_related(self, article: Dict) -> bool:
        """Check if article is related to AI based on keywords"""
        text_to_check = ' '.join([
            article.get('title', ''),
            article.get('summary', ''),
            ' '.join(article.get('tags', []))
        ]).lower()
        
        return any(keyword in text_to_check for keyword in self.keywords)
    
    def scrape_all_feeds(self) -> List[Dict]:
        """Scrape all configured RSS feeds"""
        all_articles = []
        
        for feed_config in self.config.get('rss_feeds', []):
            articles = self.parse_feed(feed_config)
            all_articles.extend(articles)
            logger.info(f"Found {len(articles)} AI articles from {feed_config['name']}")
            
        return all_articles
    
    def get_feed_info(self, feed_url: str) -> Dict:
        """Get information about an RSS feed"""
        try:
            feed = feedparser.parse(feed_url)
            
            feed_metadata: Any = feed.feed
            info = {
                'title': feed_metadata.get('title', 'Unknown'),
                'description': feed_metadata.get('description', ''),
                'link': feed_metadata.get('link', ''),
                'language': feed_metadata.get('language', ''),
                'entries_count': len(feed.entries),
                'last_updated': None
            }
            
            if hasattr(feed_metadata, 'updated_parsed'):
                info['last_updated'] = datetime(*feed_metadata.updated_parsed[:6]).isoformat()
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting feed info for {feed_url}: {str(e)}")
            return {}
    
    def _validate_feed(self, feed_url: str) -> bool:
        """Validate RSS feed URL and basic accessibility"""
        try:
            import requests
            
            # Basic URL validation
            parsed = urlparse(feed_url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Try to fetch the feed with a quick HEAD request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.head(feed_url, timeout=15, allow_redirects=True, headers=headers)
            
            # Check if response is successful
            if response.status_code != 200:
                logger.debug(f"RSS feed returned status {response.status_code}: {feed_url}")
                return False
            
            # Check content type (should be XML-related)
            content_type = response.headers.get('content-type', '').lower()
            valid_types = ['application/rss+xml', 'application/xml', 'text/xml', 
                          'application/atom+xml', 'text/rss+xml']
            
            if not any(valid_type in content_type for valid_type in valid_types):
                # Some feeds don't set proper content-type, so this is just a warning
                logger.debug(f"Unexpected content-type for RSS feed {feed_url}: {content_type}")
            
            return True
            
        except Exception as e:
            logger.warning(f"Feed validation failed for {feed_url}: {str(e)}")
            return False
    
    def test_all_feeds(self) -> Dict[str, bool]:
        """Test all configured RSS feeds and return validation results"""
        results = {}
        
        logger.info("Testing all RSS feeds for validity...")
        
        for feed_config in self.config.get('rss_feeds', []):
            feed_name = feed_config['name']
            feed_url = feed_config['url']
            
            is_valid = self._validate_feed(feed_url)
            results[feed_name] = is_valid
            
            if is_valid:
                logger.info(f"✅ {feed_name}: Valid")
            else:
                logger.warning(f"❌ {feed_name}: Invalid or inaccessible")
        
        # Summary
        valid_count = sum(results.values())
        total_count = len(results)
        logger.info(f"RSS Feed validation complete: {valid_count}/{total_count} feeds are valid")
        
        return results