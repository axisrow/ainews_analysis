import re
import json
from datetime import datetime, timedelta
from typing import Optional
from bs4 import BeautifulSoup
import logging
from dateutil import parser as date_parser

logger = logging.getLogger(__name__)


class DateExtractor:
    """Enhanced date extraction for web articles"""
    
    def __init__(self, max_age_days: int = 90):
        self.max_age_days = max_age_days
        self.now = datetime.now()
        self.min_date = self.now - timedelta(days=max_age_days)
        
        # Common date patterns in URLs
        self.url_date_patterns = [
            r'/(\d{4})/(\d{1,2})/(\d{1,2})/',  # /2024/12/19/
            r'/(\d{4})-(\d{1,2})-(\d{1,2})/',  # /2024-12-19/
            r'/(\d{8})/',                       # /20241219/
            r'date=(\d{4}-\d{1,2}-\d{1,2})',   # ?date=2024-12-19
        ]
    
    def extract_date(self, url: str, soup: Optional[BeautifulSoup] = None) -> Optional[datetime]:
        """Extract publication date from URL and/or HTML content"""
        
        # Try URL-based extraction first
        date = self._extract_from_url(url)
        if date:
            return date
            
        # If no soup provided, we can only use URL
        if not soup:
            return None
            
        # Try various HTML-based methods
        methods = [
            self._extract_from_json_ld,
            self._extract_from_meta_tags,
            self._extract_from_time_elements,
            self._extract_from_microdata,
            self._extract_from_content
        ]
        
        for method in methods:
            try:
                date = method(soup)
                if date:
                    logger.debug(f"Date extracted using {method.__name__}: {date}")
                    return date
            except Exception as e:
                logger.debug(f"Date extraction method {method.__name__} failed: {e}")
                continue
                
        return None
    
    def _extract_from_url(self, url: str) -> Optional[datetime]:
        """Extract date from URL patterns"""
        try:
            for pattern in self.url_date_patterns:
                match = re.search(pattern, url)
                if match:
                    if len(match.groups()) == 3:
                        year, month, day = match.groups()
                        date = datetime(int(year), int(month), int(day))
                    elif len(match.groups()) == 1:
                        date_str = match.group(1)
                        if len(date_str) == 8:  # YYYYMMDD
                            year = int(date_str[:4])
                            month = int(date_str[4:6])
                            day = int(date_str[6:8])
                            date = datetime(year, month, day)
                        else:  # ISO format
                            date = date_parser.parse(date_str)
                    
                    if self._is_valid_date(date):
                        return date
        except Exception as e:
            logger.debug(f"URL date extraction failed: {e}")
            
        return None
    
    def _extract_from_json_ld(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract date from JSON-LD structured data"""
        scripts = soup.find_all('script', type='application/ld+json')
        
        for script in scripts:
            try:
                data = json.loads(script.string)
                
                # Handle array of objects
                if isinstance(data, list):
                    data = data[0] if data else {}
                
                # Look for article-specific date fields
                date_fields = [
                    'datePublished', 'dateCreated', 'dateModified',
                    'publishedDate', 'createdDate', 'modifiedDate'
                ]
                
                for field in date_fields:
                    if field in data:
                        date = self._parse_date_string(data[field])
                        if date:
                            return date
                            
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.debug(f"JSON-LD parsing failed: {e}")
                continue
                
        return None
    
    def _extract_from_meta_tags(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract date from meta tags"""
        meta_selectors = [
            # Open Graph
            'meta[property="article:published_time"]',
            'meta[property="article:modified_time"]',
            'meta[property="og:published_time"]',
            
            # Twitter Cards
            'meta[name="twitter:published_time"]',
            'meta[name="twitter:created_at"]',
            
            # Generic meta tags
            'meta[name="date"]',
            'meta[name="publish-date"]',
            'meta[name="publication-date"]',
            'meta[name="article:published"]',
            'meta[name="created"]',
            'meta[name="pubdate"]',
            'meta[name="DC.date"]',
            'meta[name="DC.created"]',
            'meta[name="DC.date.created"]',
            
            # Schema.org
            'meta[itemprop="datePublished"]',
            'meta[itemprop="dateCreated"]',
            'meta[itemprop="dateModified"]',
        ]
        
        for selector in meta_selectors:
            meta = soup.select_one(selector)
            if meta:
                content = meta.get('content')
                if content:
                    date = self._parse_date_string(content)
                    if date:
                        return date
                        
        return None
    
    def _extract_from_time_elements(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract date from HTML time elements"""
        time_elements = soup.find_all('time')
        
        for time_elem in time_elements:
            # Check datetime attribute first
            datetime_attr = time_elem.get('datetime')
            if datetime_attr:
                date = self._parse_date_string(datetime_attr)
                if date:
                    return date
            
            # Check pubdate attribute
            if time_elem.get('pubdate'):
                text = time_elem.get_text(strip=True)
                date = self._parse_date_string(text)
                if date:
                    return date
                    
        return None
    
    def _extract_from_microdata(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract date from microdata attributes"""
        microdata_selectors = [
            '[itemprop="datePublished"]',
            '[itemprop="dateCreated"]',
            '[itemprop="dateModified"]',
            '[itemprop="publishedDate"]',
        ]
        
        for selector in microdata_selectors:
            elements = soup.select(selector)
            for elem in elements:
                # Check content attribute first
                content = elem.get('content')
                if content:
                    date = self._parse_date_string(content)
                    if date:
                        return date
                
                # Check datetime attribute
                datetime_attr = elem.get('datetime')
                if datetime_attr:
                    date = self._parse_date_string(datetime_attr)
                    if date:
                        return date
                
                # Check element text
                text = elem.get_text(strip=True)
                if text:
                    date = self._parse_date_string(text)
                    if date:
                        return date
                        
        return None
    
    def _extract_from_content(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract date from content using common patterns"""
        # Look for common date patterns in text
        content_selectors = [
            '.date', '.publish-date', '.published', '.post-date',
            '.article-date', '.entry-date', '.post-meta',
            '.byline', '.dateline', '.timestamp'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            for elem in elements:
                text = elem.get_text(strip=True)
                date = self._parse_date_string(text)
                if date:
                    return date
        
        # Look for date patterns in the entire content as last resort
        text_content = soup.get_text()
        date_patterns = [
            r'Published:?\s*(.+?)(?:\n|$)',
            r'Date:?\s*(.+?)(?:\n|$)',
            r'Posted:?\s*(.+?)(?:\n|$)',
            r'(\w+ \d{1,2}, \d{4})',  # "December 19, 2024"
            r'(\d{1,2}/\d{1,2}/\d{4})',  # "12/19/2024"
            r'(\d{4}-\d{1,2}-\d{1,2})',  # "2024-12-19"
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            for match in matches:
                date = self._parse_date_string(match)
                if date:
                    return date
                    
        return None
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse various date string formats"""
        if not date_str or not isinstance(date_str, str):
            return None
            
        date_str = date_str.strip()
        if not date_str:
            return None
        
        try:
            # Use dateutil parser which handles many formats
            date = date_parser.parse(date_str, fuzzy=True)
            
            # Validate the date
            if self._is_valid_date(date):
                return date
                
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Failed to parse date '{date_str}': {e}")
            
        return None
    
    def _is_valid_date(self, date: datetime) -> bool:
        """Validate if date is reasonable for a news article"""
        if not date:
            return False
            
        # Check if date is too far in the future (allow 1 day tolerance)
        if date > self.now + timedelta(days=1):
            logger.debug(f"Date too far in future: {date}")
            return False
            
        # Check if date is too old
        if date < self.min_date:
            logger.debug(f"Date too old: {date}")
            return False
            
        return True