import re
from urllib.parse import urlparse, urlunparse, parse_qs
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class URLNormalizer:
    """Utility for normalizing URLs to improve duplicate detection"""
    
    def __init__(self):
        # Parameters to remove from URLs
        self.params_to_remove = {
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            'fbclid', 'gclid', 'msclkid', 'dclid',
            'ref', 'source', 'campaign',
            '_ga', '_gid', '_gl', '_hsenc', '_hsmi',
            'mc_cid', 'mc_eid',
            'icid', 'ncid', 'cmpid',
            'share', 'shared'
        }
        
        # Common URL patterns to normalize
        self.normalization_patterns = [
            # Remove trailing index files
            (r'/index\.(html?|php)$', '/'),
            # Remove trailing slashes except for root
            (r'/$', ''),
            # Normalize www
            (r'^https?://www\.', 'https://'),
            # Normalize protocols to https
            (r'^http://', 'https://'),
        ]
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL for better duplicate detection"""
        if not url:
            return url
            
        try:
            # Parse URL
            parsed = urlparse(url.strip().lower())
            
            # Rebuild URL with normalized components
            scheme = 'https' if parsed.scheme in ('http', 'https') else parsed.scheme
            netloc = self._normalize_netloc(parsed.netloc)
            path = self._normalize_path(parsed.path)
            params = ''  # Remove params
            query = self._normalize_query(parsed.query)
            fragment = ''  # Remove fragment
            
            normalized_url = urlunparse((scheme, netloc, path, params, query, fragment))
            
            # Apply additional normalization patterns
            for pattern, replacement in self.normalization_patterns:
                normalized_url = re.sub(pattern, replacement, normalized_url)
            
            return normalized_url
            
        except Exception as e:
            logger.warning(f"Failed to normalize URL '{url}': {e}")
            return url
    
    def _normalize_netloc(self, netloc: str) -> str:
        """Normalize network location (domain)"""
        if not netloc:
            return netloc
        
        # Remove www prefix
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        
        # Remove port if it's default
        if netloc.endswith(':80') or netloc.endswith(':443'):
            netloc = netloc.rsplit(':', 1)[0]
        
        return netloc
    
    def _normalize_path(self, path: str) -> str:
        """Normalize URL path"""
        if not path:
            return '/'
        
        # Remove duplicate slashes
        path = re.sub(r'/+', '/', path)
        
        # Remove trailing slash unless it's root
        if len(path) > 1 and path.endswith('/'):
            path = path[:-1]
        
        return path
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query parameters by removing tracking parameters"""
        if not query:
            return ''
        
        try:
            # Parse query parameters
            params = parse_qs(query, keep_blank_values=False)
            
            # Remove tracking parameters
            filtered_params = {
                key: value for key, value in params.items()
                if key.lower() not in self.params_to_remove
            }
            
            # Rebuild query string
            if not filtered_params:
                return ''
            
            # Sort parameters for consistency
            sorted_params = sorted(filtered_params.items())
            query_parts = []
            
            for key, values in sorted_params:
                for value in sorted(values):
                    query_parts.append(f"{key}={value}")
            
            return '&'.join(query_parts)
            
        except Exception as e:
            logger.warning(f"Failed to normalize query '{query}': {e}")
            return query
    
    def get_canonical_url(self, url: str, canonical_url: Optional[str] = None) -> str:
        """Get canonical URL, preferring provided canonical URL"""
        if canonical_url:
            normalized_canonical = self.normalize_url(canonical_url)
            if normalized_canonical:
                return normalized_canonical
        
        return self.normalize_url(url)
    
    def are_urls_equivalent(self, url1: str, url2: str) -> bool:
        """Check if two URLs are equivalent after normalization"""
        return self.normalize_url(url1) == self.normalize_url(url2)