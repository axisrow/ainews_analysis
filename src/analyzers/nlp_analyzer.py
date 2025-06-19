import spacy
import nltk
from typing import List, Dict, Optional, Tuple
from collections import Counter
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import numpy as np

logger = logging.getLogger(__name__)


class NLPAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        
        # Load spaCy model
        self.model_name = config.get('spacy_model', 'en_core_web_sm')
        try:
            self.nlp = spacy.load(self.model_name)
        except OSError:
            logger.info(f"Downloading spaCy model: {self.model_name}")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', self.model_name])
            self.nlp = spacy.load(self.model_name)
        
        # Download NLTK data if needed
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        
        # Initialize vectorizer for topic modeling
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def analyze_article(self, article: Dict) -> Dict:
        """Perform comprehensive NLP analysis on a single article"""
        text = f"{article.get('title', '')} {article.get('summary', '')}"
        
        if not text.strip():
            return article
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract entities
        article['entities'] = self._extract_entities(doc)
        
        # Extract key phrases
        article['key_phrases'] = self._extract_key_phrases(doc)
        
        # Basic text statistics
        article['word_count'] = len([token for token in doc if not token.is_punct])
        article['sentence_count'] = len(list(doc.sents))
        
        # Extract main topics/themes
        article['themes'] = self._extract_themes(text)
        
        return article
    
    def _extract_entities(self, doc) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ in self.config.get('ner', {}).get('entities', []):
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                entities[ent.label_].append(ent.text)
        
        # Deduplicate while preserving order
        for label in entities:
            entities[label] = list(dict.fromkeys(entities[label]))
            
        return entities
    
    def _extract_key_phrases(self, doc, num_phrases: int = 5) -> List[str]:
        """Extract key phrases using noun chunks and custom logic"""
        # Extract noun chunks
        chunks = [chunk.text.lower() for chunk in doc.noun_chunks 
                 if len(chunk.text.split()) <= 3]
        
        # Extract important bigrams/trigrams
        tokens = [token.lemma_.lower() for token in doc 
                 if not token.is_stop and not token.is_punct and len(token.text) > 2]
        
        # Count frequencies
        phrase_freq = Counter(chunks)
        
        # Get top phrases
        top_phrases = [phrase for phrase, _ in phrase_freq.most_common(num_phrases)]
        
        return top_phrases
    
    def _extract_themes(self, text: str) -> List[str]:
        """Extract main themes using predefined AI-related categories"""
        themes = []
        text_lower = text.lower()
        
        # Define theme keywords
        theme_keywords = {
            'Machine Learning': ['machine learning', 'ml', 'supervised', 'unsupervised', 
                               'training', 'model', 'algorithm'],
            'Deep Learning': ['deep learning', 'neural network', 'cnn', 'rnn', 'transformer',
                            'backpropagation'],
            'Natural Language Processing': ['nlp', 'natural language', 'text analysis', 
                                          'language model', 'chatbot'],
            'Computer Vision': ['computer vision', 'image recognition', 'object detection',
                              'facial recognition', 'image processing'],
            'AI Ethics': ['ethics', 'bias', 'fairness', 'responsible ai', 'ai safety',
                         'alignment', 'transparency'],
            'AI Regulation': ['regulation', 'policy', 'governance', 'compliance', 'law',
                            'legal', 'government'],
            'Generative AI': ['generative', 'gpt', 'chatgpt', 'dalle', 'midjourney',
                            'stable diffusion', 'llm', 'large language model'],
            'AI Applications': ['healthcare', 'finance', 'education', 'autonomous',
                              'robotics', 'automation']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                themes.append(theme)
                
        return themes
    
    def analyze_corpus(self, articles: List[Dict]) -> Dict:
        """Analyze entire corpus of articles for trends and patterns"""
        if not articles:
            return {}
        
        # Combine all text
        corpus = [f"{a.get('title', '')} {a.get('summary', '')}" 
                 for a in articles]
        
        # Topic modeling
        topics = self._perform_topic_modeling(corpus)
        
        # Entity frequency analysis
        all_entities = self._aggregate_entities(articles)
        
        # Theme distribution
        theme_dist = self._analyze_theme_distribution(articles)
        
        # Temporal analysis
        temporal_trends = self._analyze_temporal_trends(articles)
        
        return {
            'topics': topics,
            'top_entities': all_entities,
            'theme_distribution': theme_dist,
            'temporal_trends': temporal_trends,
            'corpus_size': len(articles)
        }
    
    def _perform_topic_modeling(self, corpus: List[str], num_topics: int = None) -> List[Dict]:
        """Perform topic modeling on corpus"""
        if num_topics is None:
            num_topics = self.config.get('topics', {}).get('num_topics', 10)
            
        try:
            # Vectorize documents
            doc_term_matrix = self.vectorizer.fit_transform(corpus)
            
            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Perform topic modeling
            method = self.config.get('topics', {}).get('method', 'lda')
            
            if method == 'lda':
                model = LatentDirichletAllocation(
                    n_components=num_topics,
                    random_state=42
                )
            else:  # NMF
                model = NMF(n_components=num_topics, random_state=42)
                
            model.fit(doc_term_matrix)
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words[:5],  # Top 5 words
                    'weight': float(topic[top_words_idx].mean())
                })
                
            return topics
            
        except Exception as e:
            logger.error(f"Error in topic modeling: {str(e)}")
            return []
    
    def _aggregate_entities(self, articles: List[Dict]) -> Dict[str, List[Tuple[str, int]]]:
        """Aggregate and count entities across all articles"""
        entity_counts = {}
        
        for article in articles:
            entities = article.get('entities', {})
            if entities and isinstance(entities, dict):
                for ent_type, ent_list in entities.items():
                    if ent_type not in entity_counts:
                        entity_counts[ent_type] = Counter()
                    if isinstance(ent_list, list):
                        entity_counts[ent_type].update(ent_list)
        
        # Get top entities per type
        top_entities = {}
        for ent_type, counter in entity_counts.items():
            top_entities[ent_type] = counter.most_common(10)
            
        return top_entities
    
    def _analyze_theme_distribution(self, articles: List[Dict]) -> Dict[str, int]:
        """Analyze distribution of themes across articles"""
        theme_counts = Counter()
        
        for article in articles:
            themes = article.get('themes', [])
            if themes and isinstance(themes, list):
                theme_counts.update(themes)
            
        return dict(theme_counts)
    
    def _analyze_temporal_trends(self, articles: List[Dict]) -> Dict:
        """Analyze how topics change over time"""
        # This is a placeholder for temporal analysis
        # In a full implementation, you would track theme/entity changes over time
        return {
            'trend_analysis': 'Not implemented yet',
            'recommendation': 'Group articles by date and analyze theme shifts'
        }