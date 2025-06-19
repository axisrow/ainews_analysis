import praw
import requests
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging
import hashlib
from praw.models import Comment

logger = logging.getLogger(__name__)


class RedditScraper:
    def __init__(self, config: Dict):
        self.config = config
        self.keywords = [kw.lower() for kw in config.get('search_keywords', [])]
        
        # Initialize Reddit instance (read-only mode, no credentials needed)
        try:
            self.reddit = praw.Reddit(
                client_id="dummy",
                client_secret="dummy", 
                user_agent=config.get('user_agent', 'AI News Research Bot 1.0')
            )
            self.reddit.read_only = True
        except Exception:
            logger.warning("PRAW initialization failed, will use requests fallback")
            self.reddit = None
        
    def scrape_subreddit(self, subreddit_name: str) -> List[Dict]:
        """Scrape a single subreddit for AI-related posts - DISABLED due to API changes"""
        # Reddit scraping is disabled due to API authentication requirements since 2023
        logger.info(f"Reddit scraping is disabled for r/{subreddit_name} due to API authentication requirements")
        return []
    
    def _extract_post_info(self, post, subreddit_name: str) -> Optional[Dict]:
        """Extract article information from Reddit post"""
        try:
            article = {
                'source': f"Reddit - r/{subreddit_name}",
                'source_type': 'reddit',
                'scraped_at': datetime.now().isoformat(),
                'id': post.id,
                'title': post.title,
                'url': f"https://reddit.com{post.permalink}",
                'reddit_url': post.url,  # External URL if link post
                'summary': post.selftext[:500] if post.selftext else '',
                'author': str(post.author) if post.author else '[deleted]',
                'published_date': datetime.fromtimestamp(post.created_utc).isoformat(),
                'score': post.score,
                'upvote_ratio': post.upvote_ratio,
                'num_comments': post.num_comments,
                'post_type': 'text' if post.is_self else 'link',
                'tags': [] # Initialize tags as an empty list
            }
            
            if post.link_flair_text:
                article['tags'].append(post.link_flair_text)
            
            return article
            
        except Exception as e:
            logger.warning(f"Error extracting post info: {str(e)}")
            return None
    
    def _is_ai_related(self, article: Dict) -> bool:
        """Check if post is related to AI based on keywords"""
        text_to_check = ' '.join([
            article.get('title', ''),
            article.get('summary', ''),
            ' '.join(article.get('tags', []))
        ]).lower()
        
        return any(keyword in text_to_check for keyword in self.keywords)
    
    def scrape_all_subreddits(self) -> List[Dict]:
        """Scrape all configured subreddits - DISABLED due to API changes"""
        if not self.config.get('enabled', True):
            logger.info("Reddit scraping is disabled in configuration")
            return []
        
        # Even if enabled in old configs, disable due to API changes
        logger.info("Reddit scraping is disabled due to API authentication requirements since 2023")
        return []
    
    def search_reddit(self, query: str, limit: int = 100) -> List[Dict]:
        """Search across all of Reddit for AI-related posts"""
        articles = []
        
        if not self.reddit:
            logger.warning("Reddit instance not initialized. Cannot search Reddit.")
            return []

        try:
            logger.info(f"Searching Reddit for: {query}")
            
            for post in self.reddit.subreddit('all').search(query, limit=limit, sort='relevance'):
                article = self._extract_post_info(post, str(post.subreddit))
                
                if article:
                    articles.append(article)
                    
        except Exception as e:
            logger.error(f"Error searching Reddit: {str(e)}")
            
        return articles
    
    def get_post_comments(self, post_id: str, limit: int = 10) -> List[Dict]:
        """Get top comments from a Reddit post"""
        comments = []
        
        if not self.reddit:
            logger.warning(f"Reddit instance not initialized. Cannot get comments for post {post_id}.")
            return []

        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)
            
            for comment in submission.comments.list()[:limit]:
                # Ensure the comment is a Comment object and not MoreComments
                if isinstance(comment, Comment) and hasattr(comment, 'body'):
                    comments.append({
                        'id': comment.id,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'body': comment.body,
                        'score': comment.score,
                        'created_at': datetime.fromtimestamp(comment.created_utc).isoformat()
                    })
                    
        except Exception as e:
            logger.error(f"Error getting comments for post {post_id}: {str(e)}")
            
        return comments
    
    def _scrape_subreddit_requests(self, subreddit_name: str) -> List[Dict]:
        """Fallback method using requests to scrape Reddit"""
        articles = []
        
        try:
            
            # Build URL based on sort method
            sort_by = self.config.get('sort_by', 'hot')
            time_filter = self.config.get('time_filter', 'week')
            
            if sort_by == 'top':
                url = f"https://www.reddit.com/r/{subreddit_name}/top/.json?t={time_filter}"
            elif sort_by == 'new':
                url = f"https://www.reddit.com/r/{subreddit_name}/new/.json"
            else:
                url = f"https://www.reddit.com/r/{subreddit_name}/hot/.json"
            headers = {
                'User-Agent': 'AI News Research Bot 1.0'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            posts = data.get('data', {}).get('children', [])
            
            limit = self.config.get('limit', 100)
            for post_data in posts[:limit]:  # Use configured limit
                post = post_data.get('data', {})
                
                article = {
                    'id': post.get('id'),
                    'title': post.get('title'),
                    'url': f"https://reddit.com{post.get('permalink')}",
                    'reddit_url': post.get('url'),
                    'summary': post.get('selftext', '')[:500],
                    'author': post.get('author', '[deleted]'),
                    'published_date': datetime.fromtimestamp(post.get('created_utc', 0)),
                    'upvote_ratio': post.get('upvote_ratio', 0),
                    'source': f"Reddit - r/{subreddit_name}",
                    'source_type': 'reddit',
                    'scraped_at': datetime.now(),
                    'post_type': 'text' if post.get('is_self') else 'link',
                    'reddit_score': post.get('score', 0),
                    'reddit_comments': post.get('num_comments', 0)
                }
                
                # Add flair as tags
                if post.get('link_flair_text'):
                    article['tags'] = [post.get('link_flair_text')]
                
                if self._is_ai_related(article):
                    articles.append(article)
                    
        except Exception as e:
            logger.error(f"Error in requests fallback for r/{subreddit_name}: {str(e)}")
            
        return articles