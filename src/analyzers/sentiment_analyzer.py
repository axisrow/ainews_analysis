from transformers import pipeline
from textblob import TextBlob
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize transformer-based sentiment analyzer
        model_name = config.get('sentiment', {}).get(
            'model', 
            'cardiffnlp/twitter-roberta-base-sentiment'
        )
        
        try:
            self.transformer_analyzer = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name
            )
            self.use_transformer = True
        except Exception as e:
            logger.warning(f"Could not load transformer model: {str(e)}")
            logger.info("Falling back to TextBlob for sentiment analysis")
            self.use_transformer = False
            
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of a single text"""
        if not text or not text.strip():
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
            }
        
        if self.use_transformer:
            return self._transformer_sentiment(text)
        else:
            return self._textblob_sentiment(text)
    
    def _transformer_sentiment(self, text: str) -> Dict:
        """Use transformer model for sentiment analysis"""
        try:
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            raw_results = self.transformer_analyzer(text)
            if isinstance(raw_results, list) and raw_results:
                results = raw_results[0]
            else:
                logger.error(f"Transformer sentiment analysis returned unexpected result: {raw_results}. Falling back to TextBlob.")
                return self._textblob_sentiment(text)
            
            # Map labels to standard format
            label_map = {
                'POSITIVE': 'positive',
                'NEGATIVE': 'negative',
                'NEUTRAL': 'neutral',
                'LABEL_0': 'negative',  # For some models
                'LABEL_1': 'neutral',
                'LABEL_2': 'positive'
            }
            
            sentiment = label_map.get(results['label'], results['label'].lower())
            
            return {
                'sentiment': sentiment,
                'confidence': results['score'],
                'scores': {sentiment: results['score']},
                'method': 'transformer'
            }
            
        except Exception as e:
            logger.error(f"Error in transformer sentiment analysis: {str(e)}")
            return self._textblob_sentiment(text)
    
    def _textblob_sentiment(self, text: str) -> Dict:
        """Use TextBlob for sentiment analysis"""
        try:
            blob = TextBlob(text)
            sentiment_obj = blob.sentiment
            polarity = sentiment_obj.polarity  # type: ignore
            subjectivity = sentiment_obj.subjectivity  # type: ignore
            
            # Convert polarity to sentiment label
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Create pseudo-confidence score
            confidence = abs(polarity)
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'scores': {
                    'positive': max(0, polarity),
                    'negative': abs(min(0, polarity)),
                    'neutral': 1 - abs(polarity)
                },
                'method': 'textblob'
            }
            
        except Exception as e:
            logger.error(f"Error in TextBlob sentiment analysis: {str(e)}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
                'error': str(e)
            }
    
    def analyze_article_sentiment(self, article: Dict) -> Dict:
        """Analyze sentiment for an article"""
        # Combine title and summary for analysis
        text = f"{article.get('title', '')} {article.get('summary', '')}"
        
        sentiment_result = self.analyze_sentiment(text)
        
        # Add sentiment to article
        article['sentiment'] = sentiment_result
        
        # Analyze sentiment aspects if content is available
        if 'content' in article and article['content']:
            article['sentiment_aspects'] = self._analyze_aspects(article['content'])
            
        return article
    
    def _analyze_aspects(self, text: str) -> Dict:
        """Analyze sentiment for different aspects of AI"""
        aspects = {
            'innovation': ['breakthrough', 'innovation', 'advance', 'progress', 'development'],
            'concerns': ['concern', 'risk', 'danger', 'threat', 'problem', 'issue'],
            'benefits': ['benefit', 'advantage', 'improve', 'help', 'assist', 'enhance'],
            'ethics': ['ethical', 'moral', 'responsible', 'bias', 'fairness', 'transparency'],
            'regulation': ['regulation', 'policy', 'law', 'govern', 'control', 'restrict']
        }
        
        aspect_sentiments = {}
        
        for aspect, keywords in aspects.items():
            # Extract sentences containing aspect keywords
            sentences = []
            for sentence in text.split('.'):
                if any(keyword in sentence.lower() for keyword in keywords):
                    sentences.append(sentence)
            
            if sentences:
                # Analyze sentiment of aspect-related sentences
                aspect_text = ' '.join(sentences)
                aspect_sentiments[aspect] = self.analyze_sentiment(aspect_text)
                
        return aspect_sentiments
    
    def analyze_corpus_sentiment(self, articles: List[Dict]) -> Dict:
        """Analyze sentiment trends across corpus"""
        sentiments = []
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for article in articles:
            if 'sentiment' in article and article['sentiment'] and isinstance(article['sentiment'], dict):
                sentiment = article['sentiment'].get('sentiment', 'neutral')
                sentiments.append(sentiment)
                sentiment_counts[sentiment] += 1
        
        # Calculate percentages
        total = len(sentiments)
        sentiment_distribution = {
            k: (v / total * 100) if total > 0 else 0 
            for k, v in sentiment_counts.items()
        }
        
        # Analyze sentiment by source
        sentiment_by_source = self._sentiment_by_source(articles)
        
        # Analyze sentiment over time
        sentiment_timeline = self._sentiment_timeline(articles)
        
        return {
            'overall_distribution': sentiment_distribution,
            'counts': sentiment_counts,
            'by_source': sentiment_by_source,
            'timeline': sentiment_timeline,
            'total_articles': total
        }
    
    def _sentiment_by_source(self, articles: List[Dict]) -> Dict:
        """Analyze sentiment distribution by news source"""
        source_sentiments = {}
        
        for article in articles:
            source = article.get('source', 'Unknown')
            sentiment_data = article.get('sentiment', {})
            if sentiment_data and isinstance(sentiment_data, dict):
                sentiment = sentiment_data.get('sentiment', 'neutral')
            else:
                sentiment = 'neutral'
            
            if source not in source_sentiments:
                source_sentiments[source] = {
                    'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0
                }
            
            source_sentiments[source][sentiment] += 1
            source_sentiments[source]['total'] += 1
        
        # Convert to percentages
        for source, counts in source_sentiments.items():
            total = counts['total']
            if total > 0:
                for sentiment in ['positive', 'negative', 'neutral']:
                    counts[f'{sentiment}_pct'] = counts[sentiment] / total * 100
                    
        return source_sentiments
    
    def _sentiment_timeline(self, articles: List[Dict]) -> List[Dict]:
        """Analyze sentiment changes over time"""
        # Group articles by date
        timeline = {}
        
        for article in articles:
            published_date = article.get('published_date', '')
            date = published_date[:10] if published_date else ''  # Get date part only
            if date and 'sentiment' in article and article['sentiment'] and isinstance(article['sentiment'], dict):
                if date not in timeline:
                    timeline[date] = {
                        'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0
                    }
                
                sentiment = article['sentiment'].get('sentiment', 'neutral')
                timeline[date][sentiment] += 1
                timeline[date]['total'] += 1
        
        # Convert to list and sort by date
        timeline_list = []
        for date, counts in timeline.items():
            entry = {'date': date, **counts}
            # Calculate sentiment score (-1 to 1)
            if counts['total'] > 0:
                entry['sentiment_score'] = (
                    counts['positive'] - counts['negative']
                ) / counts['total']
            else:
                entry['sentiment_score'] = 0
            timeline_list.append(entry)
        
        timeline_list.sort(key=lambda x: x['date'])
        
        return timeline_list