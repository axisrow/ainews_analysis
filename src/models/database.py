from sqlalchemy import create_engine, Column, String, Text, DateTime, Float, Integer, JSON, Boolean, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import logging
from typing import List, Dict, Optional
import os
from dateutil import parser as date_parser
import sys
from pathlib import Path

from src.utils.url_normalizer import URLNormalizer

logger = logging.getLogger(__name__)

Base = declarative_base()


class Article(Base):
    __tablename__ = 'articles'
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    url = Column(String, unique=True, nullable=False)
    source = Column(String)
    source_type = Column(String)  # 'web', 'rss', 'reddit'
    summary = Column(Text)
    content = Column(Text)
    author = Column(String)
    published_date = Column(DateTime)
    scraped_at = Column(DateTime, default=datetime.now)
    
    # Analysis results
    sentiment = Column(JSON)  # Sentiment analysis results
    entities = Column(JSON)   # Named entities
    themes = Column(JSON)     # Extracted themes
    key_phrases = Column(JSON)  # Key phrases
    
    # Metrics
    word_count = Column(Integer)
    sentence_count = Column(Integer)
    
    # Reddit specific
    reddit_score = Column(Integer)
    reddit_comments = Column(Integer)
    upvote_ratio = Column(Float)
    reddit_url = Column(String)  # External URL from Reddit post
    post_type = Column(String)   # 'text' or 'link' for Reddit posts
    
    # Additional metadata
    tags = Column(JSON)
    
    # Analysis status
    analyzed = Column(Boolean, default=False)  # Flag to track if article has been analyzed
    genai_analysis = Column(JSON)  # GenAI analysis results
    
    # Date extraction status
    needs_manual_review = Column(Boolean, default=False)  # Flag for articles needing manual review
    date_extraction_attempted = Column(Boolean, default=False)  # Flag to track date extraction attempts
    
    def to_dict(self) -> Dict:
        """Convert article to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'url': self.url,
            'source': self.source,
            'source_type': self.source_type,
            'summary': self.summary,
            'content': self.content,
            'author': self.author,
            'published_date': self.published_date.isoformat() if self.published_date is not None else None,
            'scraped_at': self.scraped_at.isoformat() if self.scraped_at is not None else None,
            'sentiment': self.sentiment,
            'entities': self.entities,
            'themes': self.themes,
            'key_phrases': self.key_phrases,
            'word_count': self.word_count,
            'sentence_count': self.sentence_count,
            'reddit_score': self.reddit_score,
            'reddit_comments': self.reddit_comments,
            'upvote_ratio': self.upvote_ratio,
            'reddit_url': self.reddit_url,
            'post_type': self.post_type,
            'tags': self.tags,
            'analyzed': self.analyzed,
            'genai_analysis': self.genai_analysis,
            'needs_manual_review': self.needs_manual_review,
            'date_extraction_attempted': self.date_extraction_attempted
        }


class Database:
    def __init__(self, config: Dict):
        self.config = config
        self.url_normalizer = URLNormalizer()
        
        # Create database directory if it doesn't exist
        db_path = config.get('path', 'data/ai_news.db')
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        
        # Create engine
        if config.get('type') == 'sqlite':
            self.engine = create_engine(f'sqlite:///{db_path}')
        else:
            # PostgreSQL support (if needed later)
            raise NotImplementedError("Only SQLite is supported currently")
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Run migrations for existing databases
        self._run_migrations()
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def _run_migrations(self):
        """Run database migrations for schema updates"""
        try:
            # Check if analyzed column exists
            with self.engine.connect() as conn:
                result = conn.execute(text("PRAGMA table_info(articles)"))
                columns = [row[1] for row in result.fetchall()]
                
                # Add analyzed column if missing
                if 'analyzed' not in columns:
                    logger.info("Adding 'analyzed' column to articles table...")
                    conn.execute(text("ALTER TABLE articles ADD COLUMN analyzed BOOLEAN DEFAULT FALSE"))
                    conn.commit()
                
                # Add genai_analysis column if missing
                if 'genai_analysis' not in columns:
                    logger.info("Adding 'genai_analysis' column to articles table...")
                    conn.execute(text("ALTER TABLE articles ADD COLUMN genai_analysis JSON"))
                    conn.commit()
                
                # Add needs_manual_review column if missing
                if 'needs_manual_review' not in columns:
                    logger.info("Adding 'needs_manual_review' column to articles table...")
                    conn.execute(text("ALTER TABLE articles ADD COLUMN needs_manual_review BOOLEAN DEFAULT FALSE"))
                    conn.commit()
                
                # Add date_extraction_attempted column if missing
                if 'date_extraction_attempted' not in columns:
                    logger.info("Adding 'date_extraction_attempted' column to articles table...")
                    conn.execute(text("ALTER TABLE articles ADD COLUMN date_extraction_attempted BOOLEAN DEFAULT FALSE"))
                    conn.commit()
                    
                logger.info("Database migrations completed successfully")
                
        except Exception as e:
            logger.warning(f"Migration failed (this may be normal for new databases): {e}")
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def close(self):
        """Close database connections"""
        if hasattr(self, 'engine'):
            self.engine.dispose()
    
    def _convert_datetime_fields(self, article_data: Dict) -> Dict:
        """Convert string datetime fields to datetime objects"""
        data = article_data.copy()
        datetime_fields = ['published_date', 'scraped_at']
        
        for field in datetime_fields:
            if field in data and data[field]:
                value = data[field]
                if isinstance(value, str):
                    try:
                        # Try to parse ISO format datetime string
                        data[field] = date_parser.isoparse(value)
                    except (ValueError, TypeError):
                        # If parsing fails, set to None
                        logger.warning(f"Could not parse datetime field {field}: {value}")
                        data[field] = None
                elif not isinstance(value, datetime):
                    # If it's not string or datetime, set to None
                    data[field] = None
        
        return data
    
    def save_article(self, article_data: Dict) -> bool:
        """Save or update an article"""
        session = self.get_session()
        try:
            # Convert datetime fields from strings to datetime objects
            converted_data = self._convert_datetime_fields(article_data)
            
            # Normalize URL for better duplicate detection
            original_url = converted_data.get('url', '')
            normalized_url = self.url_normalizer.normalize_url(original_url)
            
            # Check for duplicates using both original and normalized URLs
            existing = session.query(Article).filter(
                (Article.url == original_url) | 
                (Article.url == normalized_url)
            ).first()
            
            # If no exact match, check for equivalent URLs
            if not existing and original_url != normalized_url:
                all_articles = session.query(Article).all()
                for article in all_articles:
                    if self.url_normalizer.are_urls_equivalent(original_url, article.url):
                        existing = article
                        break
            
            if existing:
                # Update existing article with new analysis data
                for key, value in converted_data.items():
                    if hasattr(existing, key) and value is not None:
                        setattr(existing, key, value)
                logger.debug(f"Updated article: {article_data.get('title', 'Unknown')[:50]}...")
                session.commit()
                return False  # Not a new article
            else:
                # Create new article
                article = Article(**converted_data)
                session.add(article)
                logger.info(f"Added new article: {converted_data.get('title', 'Unknown')[:50]}...")
                session.commit()
                return True  # New article added
            
        except Exception as e:
            logger.error(f"Error saving article: {str(e)}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def save_articles(self, articles: List[Dict]) -> int:
        """Save multiple articles"""
        saved_count = 0
        for article in articles:
            if self.save_article(article):
                saved_count += 1
        return saved_count
    
    def get_article_by_id(self, article_id: str) -> Optional[Dict]:
        """Get article by ID"""
        session = self.get_session()
        try:
            article = session.query(Article).filter_by(id=article_id).first()
            return article.to_dict() if article else None
        finally:
            session.close()
    
    def get_articles(
        self, 
        limit: int = 100, 
        offset: int = 0,
        source: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        sentiment: Optional[str] = None
    ) -> List[Dict]:
        """Get articles with filters"""
        session = self.get_session()
        try:
            query = session.query(Article)
            
            # Apply filters
            if source:
                query = query.filter(Article.source == source)
            if start_date:
                query = query.filter(Article.published_date >= start_date)
            if end_date:
                query = query.filter(Article.published_date <= end_date)
            if sentiment:
                # This requires JSON query support
                query = query.filter(
                    Article.sentiment['sentiment'].as_string() == sentiment
                )
            
            # Order by published date descending
            query = query.order_by(Article.published_date.desc())
            
            # Apply pagination
            query = query.limit(limit).offset(offset)
            
            articles = query.all()
            return [article.to_dict() for article in articles]
            
        finally:
            session.close()
    
    def get_article_count(self) -> int:
        """Get total article count"""
        session = self.get_session()
        try:
            return session.query(Article).count()
        finally:
            session.close()
    
    def get_existing_urls(self) -> set:
        """Get set of existing article URLs for duplicate checking"""
        session = self.get_session()
        try:
            urls = session.query(Article.url).all()
            return {url[0] for url in urls if url[0]}
        finally:
            session.close()
    
    def get_unanalyzed_articles(self, limit: Optional[int] = None) -> List[Dict]:
        """Get articles that haven't been analyzed yet"""
        session = self.get_session()
        try:
            query = session.query(Article).filter(Article.analyzed == False)
            if limit:
                query = query.limit(limit)
            articles = query.all()
            return [article.to_dict() for article in articles]
        finally:
            session.close()
    
    def mark_articles_analyzed(self, article_ids: List[str]):
        """Mark articles as analyzed"""
        session = self.get_session()
        try:
            session.query(Article).filter(Article.id.in_(article_ids))\
                   .update({Article.analyzed: True}, synchronize_session=False)
            session.commit()
            logger.info(f"Marked {len(article_ids)} articles as analyzed")
        except Exception as e:
            session.rollback()
            logger.error(f"Error marking articles as analyzed: {e}")
        finally:
            session.close()
    
    def get_sources(self) -> List[str]:
        """Get unique sources"""
        session = self.get_session()
        try:
            sources = session.query(Article.source).distinct().all()
            return [source[0] for source in sources if source[0]]
        finally:
            session.close()
    
    def get_latest_article_date(self) -> Optional[datetime]:
        """Get the date of the most recent article"""
        session = self.get_session()
        try:
            latest = session.query(Article.published_date)\
                          .order_by(Article.published_date.desc())\
                          .first()
            return latest[0] if latest else None
        finally:
            session.close()
    
    def search_articles(self, query: str, limit: int = 100) -> List[Dict]:
        """Search articles by text"""
        session = self.get_session()
        try:
            # Simple text search in title and summary
            articles = session.query(Article).filter(
                (Article.title.contains(query)) | 
                (Article.summary.contains(query))
            ).limit(limit).all()
            
            return [article.to_dict() for article in articles]
            
        finally:
            session.close()
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        session = self.get_session()
        try:
            total_articles = session.query(Article).count()
            
            # Count valid articles (not needing manual review)
            valid_articles = session.query(Article).filter(
                Article.needs_manual_review.isnot(True)
            ).count()
            
            # Count articles needing manual review
            manual_review_articles = session.query(Article).filter(
                Article.needs_manual_review == True
            ).count()
            
            # Source distribution
            sources = session.query(
                Article.source, 
                Article.source_type
            ).distinct().all()
            
            # Date range for ALL articles
            all_date_range = session.query(
                Article.published_date
            ).filter(Article.published_date.isnot(None)).order_by(Article.published_date).all()
            
            all_earliest = all_date_range[0][0] if all_date_range else None
            all_latest = all_date_range[-1][0] if all_date_range else None
            
            # Date range for VALID articles only
            valid_date_range = session.query(
                Article.published_date
            ).filter(
                Article.published_date.isnot(None),
                Article.needs_manual_review.isnot(True)
            ).order_by(Article.published_date).all()
            
            valid_earliest = valid_date_range[0][0] if valid_date_range else None
            valid_latest = valid_date_range[-1][0] if valid_date_range else None
            
            return {
                'total_articles': total_articles,
                'valid_articles': valid_articles,
                'manual_review_articles': manual_review_articles,
                'sources': len(sources),
                'date_range': {
                    'earliest': all_earliest.isoformat() if all_earliest else None,
                    'latest': all_latest.isoformat() if all_latest else None
                },
                'valid_date_range': {
                    'earliest': valid_earliest.isoformat() if valid_earliest else None,
                    'latest': valid_latest.isoformat() if valid_latest else None
                }
            }
            
        finally:
            session.close()