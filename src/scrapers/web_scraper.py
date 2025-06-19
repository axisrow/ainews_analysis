import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
import logging
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class WebScraper:
    def __init__(self, config: Dict):
        self.config = config
        self.session = requests.Session()
        
        # Set up headers
        headers = {
            'User-Agent': config.get('user_agent', 'Mozilla/5.0')
        }
        # Add additional headers if provided
        if 'headers' in config:
            headers.update(config['headers'])
        
        self.session.headers.update(headers)
        self.timeout = config.get('timeout', 30)
        self.delay = config.get('delay_between_requests', 2)
        self.max_retries = config.get('max_retries', 3)
        self.backoff_factor = config.get('backoff_factor', 2)
        
    def scrape_site(self, site_config: Dict) -> List[Dict]:
        """Scrape a single news site for AI articles with retry logic"""
        articles = []
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Scraping {site_config['name']} (attempt {attempt + 1}/{self.max_retries})...")
                
                response = self.session.get(
                    site_config['url'], 
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # Handle encoding properly to avoid character decode warnings
                if response.encoding is None:
                    response.encoding = 'utf-8'
                
                # Use response.text with proper encoding instead of response.content
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all article elements
                article_elements = soup.select(site_config['article_selector'])
                logger.info(f"Found {len(article_elements)} potential articles on {site_config['name']}")
                
                max_articles = self.config.get('max_articles_per_site', 50)
                for element in article_elements[:max_articles]:
                    article = self._extract_article_info(
                        element, 
                        site_config, 
                        response.url
                    )
                    
                    if article:
                        article['source'] = site_config['name']
                        article['source_type'] = 'web'
                        article['scraped_at'] = datetime.now()
                        
                        # Check if AI-related (more lenient for testing)
                        if self._is_ai_related(article):
                            articles.append(article)
                            logger.debug(f"Found AI article: {article.get('title', 'No title')[:50]}...")
                        
                    time.sleep(0.1)  # Small delay between articles
                
                # Success - break retry loop
                break
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [403, 401]:
                    logger.warning(f"{site_config['name']} returned {e.response.status_code} - site may block bots")
                    break  # Don't retry for authentication/permission errors
                elif attempt < self.max_retries - 1:
                    wait_time = self.delay * (self.backoff_factor ** attempt)
                    logger.warning(f"HTTP error for {site_config['name']}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to scrape {site_config['name']} after {self.max_retries} attempts: {e}")
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.delay * (self.backoff_factor ** attempt)
                    logger.warning(f"Request error for {site_config['name']}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to scrape {site_config['name']} after {self.max_retries} attempts: {e}")
            except Exception as e:
                logger.error(f"Unexpected error scraping {site_config['name']}: {e}")
                break
        
        logger.info(f"Collected {len(articles)} AI articles from {site_config['name']}")
        time.sleep(self.delay)  # Rate limiting between sites
        return articles
    
    def _extract_article_info(
        self, 
        element, 
        site_config: Dict, 
        base_url: str
    ) -> Optional[Dict]:
        """Extract article information from HTML element"""
        try:
            article = {}
            
            # Extract title
            title_elem = element.select_one(site_config.get('title_selector', 'h2'))
            if title_elem:
                article['title'] = title_elem.get_text(strip=True)
            
            # Extract URL
            link_elem = element.find('a')
            if link_elem and link_elem.get('href'):
                article['url'] = urljoin(base_url, link_elem['href'])
            
            # Extract summary/content preview
            content_elem = element.select_one(
                site_config.get('content_selector', 'p')
            )
            if content_elem:
                article['summary'] = content_elem.get_text(strip=True)[:500]
            
            # Generate unique ID
            if 'url' in article:
                article['id'] = hashlib.md5(
                    article['url'].encode()
                ).hexdigest()
            
            # Extract publish date if available
            date_elem = element.select_one('time')
            if date_elem:
                article['published_date'] = date_elem.get('datetime')
            
            return article if 'title' in article and 'url' in article else None
            
        except Exception as e:
            logger.warning(f"Error extracting article: {str(e)}")
            return None
    
    def _is_ai_related(self, article: Dict) -> bool:
        """Check if article is related to AI based on keywords"""
        keywords = self.config.get('search_keywords', [])
        
        text_to_check = ' '.join([
            article.get('title', ''),
            article.get('summary', '')
        ]).lower()
        
        # More lenient check - if no keywords configured, accept all
        if not keywords:
            return True
            
        is_related = any(keyword.lower() in text_to_check for keyword in keywords)
        
        if not is_related:
            logger.debug(f"Article not AI-related: {article.get('title', 'No title')[:50]}")
            
        return is_related
    
    def scrape_all_sites(self) -> List[Dict]:
        """Scrape all configured sites"""
        all_articles = []
        
        for site_config in self.config.get('sites', []):
            articles = self.scrape_site(site_config)
            all_articles.extend(articles)
            logger.info(f"Found {len(articles)} AI articles from {site_config['name']}")
            
        return all_articles
    
    def get_full_article_content(self, url: str) -> Optional[str]:
        """Fetch full article content from URL"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Handle encoding properly
            if response.encoding is None:
                response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Common article content selectors
            content_selectors = [
                'article', 
                '.article-content', 
                '.post-content',
                'main',
                '.content'
            ]
            
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    return content.get_text(separator='\n', strip=True)
                    
            # Fallback to body
            return soup.body.get_text(separator='\n', strip=True)
            
        except Exception as e:
            logger.error(f"Error fetching article content from {url}: {str(e)}")
            return None