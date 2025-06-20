"""
Keyword Analyzer for AI News Research Project
Analyzes articles in database to extract and rank keywords
"""

import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import re
import string
from pathlib import Path
import json
import yaml
from datetime import datetime, timedelta

import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class KeywordAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.nlp_config = config.get('nlp', {})
        
        # Initialize spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.warning("spaCy model not found. Download with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize NLTK stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK stopwords not found. Downloading...")
            nltk.download('stopwords')
            nltk.download('punkt_tab')
            self.stop_words = set(stopwords.words('english'))
        
        # Add comprehensive stopwords to filter out common but unhelpful terms
        self.custom_stopwords = {
            # Common news/article words
            'article', 'news', 'report', 'says', 'according', 'new', 'year', 
            'company', 'said', 'will', 'also', 'one', 'two', 'three', 'first', 'last',
            'week', 'month', 'day', 'time', 'people', 'way', 'thing', 'use',
            'make', 'get', 'go', 'see', 'know', 'think', 'want', 'give', 'post',
            'appeared', 'first', 'presenter', 'session', 'wrap', 'appeared first',
            'the post', 'post appeared', 'appeared first on',
            
            # HTML/CSS/Web terms that shouldn't be keywords
            'href', 'https', 'http', 'www', 'com', 'org', 'html', 'div', 'span',
            'class', 'style', 'img', 'src', 'alt', 'title', 'link', 'url', 'strong',
            'align', 'center', 'left', 'right', 'auto', 'margin', 'padding', 'border',
            'text', 'font', 'color', 'background', 'width', 'height', 'table', 'tbody',
            'thead', 'tfoot', 'tr', 'td', 'th', 'caption', 'container', 'cellpadding',
            'cellspacing', 'nbsp', 'br', 'li', 'ul', 'ol', 'target', 'rel', 'nofollow',
            'hljs', 'number', 'string', 'comment', 'block', 'display', 'vertical', 
            'baseline', 'transparent', 'family', 'line', 'normal', 'card', 'top',
            'em', 'pre', 'code', 'px', 'rem', 'vh', 'vw', 'rgb', 'rgba', 'hex',
            'css', 'js', 'javascript', 'jquery', 'bootstrap', 'flex', 'grid',
            
            # Generic words that add no value
            'example', 'based', 'using', 'data', 'input', 'output', 'series',
            'method', 'approach', 'technique', 'system', 'process', 'feature',
            'function', 'design', 'development', 'research', 'study', 'analysis',
            'result', 'results', 'conclusion', 'summary', 'overview', 'introduction'
        }
        self.stop_words.update(self.custom_stopwords)
        
        # Specific AI-related terms (must contain these to be considered AI-related)
        self.ai_core_terms = {
            'ai', 'artificial intelligence', 'machine learning', 'deep learning',
            'neural network', 'llm', 'transformer', 'gpt', 'bert', 'claude',
            'chatgpt', 'openai', 'anthropic', 'google ai', 'deepmind',
            'generative ai', 'nlp', 'computer vision', 'robotics', 'automation',
            'algorithm', 'model', 'training', 'dataset', 'inference', 'fine-tuning',
            'prompt engineering', 'rag', 'vector database', 'embedding',
            'multimodal', 'vision model', 'language model', 'diffusion model',
            'reinforcement learning', 'supervised learning', 'unsupervised learning',
            'perplexity', 'hugging face', 'stable diffusion', 'midjourney',
            'tensorflow', 'pytorch', 'keras', 'scikit', 'pandas', 'numpy',
            'dall', 'sora', 'midjourney', 'copilot', 'gemini', 'llama', 'cohere'
        }
        
        # Terms that are definitely NOT AI-specific (blacklist)
        self.non_ai_terms = {
            'promising results', 'show promising', 'announces breakthrough', 
            'breakthrough in', 'new developments', 'developments in',
            'technology show', 'results for', 'applications', 'breakthrough',
            'developments', 'promising', 'announces', 'show', 'technology',
            'results', 'breakthrough gpt', 'developments gpt', 'new study',
            'research shows', 'scientists discover', 'latest research',
            'study finds', 'researchers announce', 'new findings'
        }
        
    def analyze_keywords_from_database(self, database, 
                                     min_frequency: int = 3,
                                     top_n: int = 100,
                                     days_back: Optional[int] = None) -> Dict:
        """
        Analyze all articles in database to extract keywords
        
        Args:
            database: Database instance
            min_frequency: Minimum frequency for a keyword to be included
            top_n: Number of top keywords to return
            days_back: Analyze only articles from last N days (None = all)
            
        Returns:
            Dictionary with keyword analysis results
        """
        logger.info("Starting keyword analysis from database...")
        
        # Get articles from database
        session = database.get_session()
        try:
            from src.models.database import Article
            
            query = session.query(Article)
            
            # Filter by date if specified
            if days_back:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                query = query.filter(Article.published_date >= cutoff_date)
            
            articles = query.all()
            logger.info(f"Analyzing {len(articles)} articles...")
            
            if not articles:
                logger.warning("No articles found in database")
                return {
                    'status': 'no_data',
                    'message': 'No articles found in database',
                    'keywords': []
                }
            
            # Extract text from articles
            texts = []
            for article in articles:
                text_parts = []
                if article.title:
                    text_parts.append(article.title)
                if article.summary:
                    text_parts.append(article.summary)
                if article.content:
                    # Clean HTML tags from content
                    clean_content = re.sub(r'<[^>]+>', '', article.content[:1000])
                    text_parts.append(clean_content)
                    
                if text_parts:
                    # Join and clean text
                    full_text = ' '.join(text_parts)
                    # Remove URLs
                    full_text = re.sub(r'https?://\S+|www\.\S+', '', full_text)
                    texts.append(full_text)
            
            # Perform keyword extraction
            keywords = self._extract_keywords_tfidf(texts, top_n=top_n)
            entities = self._extract_named_entities(texts, min_frequency=min_frequency)
            phrases = self._extract_key_phrases(texts, min_frequency=min_frequency)
            term_frequency = self._calculate_term_frequency(texts, min_frequency=min_frequency)
            
            # Combine and rank keywords
            final_keywords = self._combine_and_rank_keywords(
                keywords, entities, phrases, term_frequency, top_n=top_n
            )
            
            # Generate statistics
            stats = {
                'total_articles': len(articles),
                'total_unique_terms': len(term_frequency),
                'date_range': {
                    'earliest': min(a.published_date for a in articles if a.published_date).isoformat() if any(a.published_date for a in articles) else None,
                    'latest': max(a.published_date for a in articles if a.published_date).isoformat() if any(a.published_date for a in articles) else None
                } if any(a.published_date for a in articles) else None
            }
            
            result = {
                'status': 'success',
                'statistics': stats,
                'keywords': final_keywords,
                'entities': entities[:50],  # Top 50 entities
                'phrases': phrases[:50],     # Top 50 phrases
                'generated_at': datetime.now().isoformat()
            }
            
            logger.info(f"Keyword analysis completed. Found {len(final_keywords)} keywords")
            return result
            
        finally:
            session.close()
    
    def _extract_keywords_tfidf(self, texts: List[str], top_n: int = 100) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF"""
        if not texts:
            return []
            
        try:
            # Configure TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=top_n * 2,
                stop_words=list(self.stop_words),
                ngram_range=(1, 3),  # Include up to 3-word phrases
                min_df=2,
                max_df=0.8
            )
            
            # Fit and transform texts
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Get feature names and scores
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            
            # Sort by score
            keyword_scores = [(feature_names[i], scores[i]) 
                            for i in scores.argsort()[::-1][:top_n]]
            
            # Boost AI-related terms
            boosted_scores = []
            for term, score in keyword_scores:
                if any(ai_term in term.lower() for ai_term in self.ai_core_terms):
                    score *= 1.5  # 50% boost for AI terms
                boosted_scores.append((term, score))
            
            # Re-sort after boosting
            boosted_scores.sort(key=lambda x: x[1], reverse=True)
            
            return boosted_scores[:top_n]
            
        except Exception as e:
            logger.error(f"Error in TF-IDF extraction: {e}")
            return []
    
    def _extract_named_entities(self, texts: List[str], min_frequency: int = 3) -> List[Tuple[str, int]]:
        """Extract named entities using spaCy"""
        if not self.nlp or not texts:
            return []
            
        entity_counts = Counter()
        
        for text in texts[:100]:  # Limit for performance
            try:
                doc = self.nlp(text[:5000])  # Limit text length
                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'PERSON', 'PRODUCT', 'GPE']:
                        entity_counts[ent.text] += 1
            except Exception as e:
                logger.debug(f"Error processing entity: {e}")
                continue
        
        # Filter by minimum frequency
        filtered_entities = [(entity, count) for entity, count in entity_counts.items() 
                           if count >= min_frequency]
        
        # Sort by frequency
        return sorted(filtered_entities, key=lambda x: x[1], reverse=True)
    
    def _extract_key_phrases(self, texts: List[str], min_frequency: int = 3) -> List[Tuple[str, int]]:
        """Extract key phrases using statistical methods"""
        phrase_counts = Counter()
        
        for text in texts:
            # Simple phrase extraction using regex
            text_lower = text.lower()
            
            # Extract 2-4 word phrases
            words = re.findall(r'\b[a-z]+\b', text_lower)
            for i in range(len(words) - 1):
                for j in range(2, min(5, len(words) - i + 1)):
                    phrase = ' '.join(words[i:i+j])
                    
                    # Filter out phrases with too many stopwords
                    phrase_words = phrase.split()
                    non_stop_words = [w for w in phrase_words if w not in self.stop_words]
                    
                    # Additional filters
                    # 1. At least 50% should be non-stopwords
                    # 2. Should contain at least one word with 3+ letters
                    # 3. Should not be just HTML/CSS terms
                    # 4. Filter out obvious HTML patterns
                    has_real_word = any(len(w) >= 3 and w.isalpha() for w in phrase_words)
                    not_html_only = not all(w in self.custom_stopwords or len(w) <= 2 for w in phrase_words)
                    
                    # Additional HTML/CSS pattern filters
                    html_patterns = ['class', 'style', 'hljs', 'div', 'span', 'display', 'align', 'line', 'vertical']
                    contains_html_pattern = any(pattern in phrase for pattern in html_patterns)
                    
                    # Filter out CSS-like phrases (e.g., "style line height", "class hljs number")
                    is_css_like = (any(w in ['style', 'class', 'hljs', 'display', 'align', 'margin', 'padding'] for w in phrase_words) and
                                   len(phrase_words) >= 2)
                    
                    # Filter out common template phrases
                    template_phrases = ['the post', 'appeared first', 'post appeared', 'presenter session']
                    is_template = any(template in phrase for template in template_phrases)
                    
                    if (len(non_stop_words) >= len(phrase_words) * 0.5 and 
                        has_real_word and not_html_only and 
                        not contains_html_pattern and not is_css_like and not is_template):
                        phrase_counts[phrase] += 1
        
        # Filter by minimum frequency
        filtered_phrases = [(phrase, count) for phrase, count in phrase_counts.items() 
                          if count >= min_frequency]
        
        return sorted(filtered_phrases, key=lambda x: x[1], reverse=True)
    
    def _calculate_term_frequency(self, texts: List[str], min_frequency: int = 3) -> Dict[str, int]:
        """Calculate simple term frequency"""
        term_counts = Counter()
        
        for text in texts:
            # Simple regex-based tokenization to avoid NLTK dependency issues
            # This matches word characters and splits on non-word characters
            tokens = re.findall(r'\b[a-z]+\b', text.lower())
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
            term_counts.update(tokens)
        
        # Filter by minimum frequency
        return {term: count for term, count in term_counts.items() if count >= min_frequency}
    
    def _combine_and_rank_keywords(self, 
                                  tfidf_keywords: List[Tuple[str, float]],
                                  entities: List[Tuple[str, int]],
                                  phrases: List[Tuple[str, int]],
                                  term_frequency: Dict[str, int],
                                  top_n: int = 100) -> List[Dict]:
        """Combine different keyword sources and create final ranking"""
        
        keyword_scores = defaultdict(float)
        keyword_sources = defaultdict(set)
        keyword_frequency = defaultdict(int)
        
        # Add TF-IDF keywords
        max_tfidf = max([score for _, score in tfidf_keywords], default=1.0)
        for term, score in tfidf_keywords:
            normalized_score = score / max_tfidf
            keyword_scores[term.lower()] += normalized_score * 1.0  # Weight: 1.0
            keyword_sources[term.lower()].add('tfidf')
        
        # Add entities
        max_entity_count = max([count for _, count in entities], default=1)
        for entity, count in entities:
            normalized_score = count / max_entity_count
            keyword_scores[entity.lower()] += normalized_score * 1.2  # Weight: 1.2 (entities are important)
            keyword_sources[entity.lower()].add('entity')
            keyword_frequency[entity.lower()] = count
        
        # Add phrases
        max_phrase_count = max([count for _, count in phrases], default=1)
        for phrase, count in phrases[:200]:  # Limit phrases
            normalized_score = count / max_phrase_count
            keyword_scores[phrase.lower()] += normalized_score * 0.8  # Weight: 0.8
            keyword_sources[phrase.lower()].add('phrase')
            keyword_frequency[phrase.lower()] = count
        
        # Add term frequency
        max_term_freq = max(term_frequency.values(), default=1)
        for term, freq in term_frequency.items():
            if term.lower() not in keyword_scores:  # Only add if not already present
                normalized_score = freq / max_term_freq
                keyword_scores[term.lower()] += normalized_score * 0.5  # Weight: 0.5
                keyword_sources[term.lower()].add('frequency')
                keyword_frequency[term.lower()] = freq
        
        # Filter out blacklisted terms and apply improved AI detection
        filtered_keywords = {}
        for keyword, score in keyword_scores.items():
            # Skip blacklisted terms
            if keyword in self.non_ai_terms:
                continue
            
            # Skip terms that are too generic
            if len(keyword) <= 2 or keyword.isdigit():
                continue
                
            filtered_keywords[keyword] = score
        
        # Remove duplicate/nested phrases
        deduplicated_keywords = self._remove_nested_phrases(filtered_keywords)
        
        # Sort by combined score
        sorted_keywords = sorted(deduplicated_keywords.items(), key=lambda x: x[1], reverse=True)
        
        # Format results with improved AI detection
        final_keywords = []
        for keyword, score in sorted_keywords[:top_n]:
            is_ai_term = self._is_ai_related_term(keyword)
            
            # Boost AI-related terms
            if is_ai_term:
                score *= 1.3
            
            final_keywords.append({
                'keyword': keyword,
                'score': round(score, 4),
                'sources': list(keyword_sources[keyword]),
                'frequency': keyword_frequency.get(keyword, 0),
                'is_ai_term': is_ai_term
            })
        
        # Re-sort after boosting
        final_keywords.sort(key=lambda x: x['score'], reverse=True)
        
        return final_keywords[:top_n]
    
    def _is_ai_related_term(self, term: str) -> bool:
        """Determine if a term is AI-related using improved logic"""
        term_lower = term.lower()
        
        # Direct match with AI core terms
        if term_lower in self.ai_core_terms:
            return True
        
        # Check if term contains any AI core terms
        for ai_term in self.ai_core_terms:
            if ai_term in term_lower:
                # Additional check: make sure it's not a generic phrase with AI term
                # For example, "results for ai" contains "ai" but is generic
                generic_patterns = ['results for', 'breakthrough in', 'developments in', 
                                   'new', 'latest', 'study', 'research', 'announces']
                
                if not any(pattern in term_lower for pattern in generic_patterns):
                    return True
        
        return False
    
    def _remove_nested_phrases(self, keywords_dict: Dict[str, float]) -> Dict[str, float]:
        """Remove phrases that are completely contained in longer phrases"""
        keywords_list = list(keywords_dict.keys())
        filtered_dict = {}
        
        for keyword in keywords_list:
            # Check if this keyword is contained in any longer keyword
            is_nested = False
            for other_keyword in keywords_list:
                if (keyword != other_keyword and 
                    len(keyword) < len(other_keyword) and 
                    keyword in other_keyword and
                    # Only remove if the longer phrase has similar or better score
                    keywords_dict[other_keyword] >= keywords_dict[keyword] * 0.8):
                    is_nested = True
                    break
            
            if not is_nested:
                filtered_dict[keyword] = keywords_dict[keyword]
        
        return filtered_dict
    
    def save_keywords(self, keywords: Dict, output_path: str = 'data/keywords.json'):
        """Save keywords to JSON file only"""
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)
        
        # Add metadata
        keywords['metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'version': '1.0',
            'editable': True,
            'description': 'Auto-generated keywords from AI news articles. Edit via dashboard.'
        }
        
        # Save as JSON only
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(keywords, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Keywords saved to {output_file}")
        return str(output_file)
    
    def load_keywords(self, keywords_file: str = 'data/keywords.json') -> Dict:
        """Load keywords from JSON file"""
        keywords_path = Path(keywords_file)
        if not keywords_path.exists():
            logger.info(f"No keywords file found at {keywords_path}")
            return {'keywords': [], 'status': 'not_found'}
        
        try:
            with open(keywords_path, 'r', encoding='utf-8') as f:
                keywords_data = json.load(f)
            
            logger.info(f"Loaded {len(keywords_data.get('keywords', []))} keywords from {keywords_path}")
            return keywords_data
        except Exception as e:
            logger.error(f"Error loading keywords from {keywords_path}: {e}")
            return {'keywords': [], 'status': 'error', 'error': str(e)}