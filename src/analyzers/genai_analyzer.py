#!/usr/bin/env python3
"""
Google GenAI analyzer for AI news content
Enhanced analysis using Google's Gemini model - Updated for new SDK
"""

import json
import time
from typing import Dict, List, Optional, Any
from loguru import logger
from google import genai
from google.genai import types


class GenAIAnalyzer:
    """AI-powered content analyzer using Google Gemini - New SDK"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize GenAI analyzer with new SDK"""
        self.config = config.get('genai', {})
        self.model_name = self.config.get('model', 'gemini-2.0-flash-001')
        self.api_key = self.config.get('api_key')
        self.max_retries = self.config.get('max_retries', 3)
        self.delay_between_requests = self.config.get('delay_between_requests', 1.0)
        self.temperature = self.config.get('temperature', 0.7)
        self.safety_settings = self.config.get('safety_settings', {})
        
        # Initialize the client if API key is available
        self.client = None
        self.fallback_count = 0
        self.api_count = 0
        
        if self.api_key and self.api_key.strip():
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"🤖 GenAI analyzer initialized with {self.model_name}")
                self._configure_safety_settings()
            except Exception as e:
                logger.warning(f"⚠️ GenAI initialization failed: {e}")
                self.client = None
        else:
            logger.info("🔑 No GOOGLE_API_KEY provided - GenAI analysis will use fallback mode")
    
    def _configure_safety_settings(self):
        """Configure safety settings for content generation"""
        self.safety_config = {
            'harm_category': ['HARM_CATEGORY_HARASSMENT', 'HARM_CATEGORY_HATE_SPEECH', 
                           'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'HARM_CATEGORY_DANGEROUS_CONTENT'],
            'threshold': self.safety_settings.get('threshold', 'BLOCK_MEDIUM_AND_ABOVE')
        }
    
    def analyze_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze article with Google Gemini using new SDK"""
        if not self.client:
            self.fallback_count += 1
            if self.fallback_count == 1:  # Log only once to avoid spam
                logger.info("📝 Using fallback GenAI analysis (no API key)")
            return self._fallback_analysis(article)
        
        try:
            # Prepare content for analysis
            content = self._prepare_content(article)
            
            # Get AI analysis with structured response
            analysis = self._get_genai_analysis_structured(content)
            
            if analysis:
                # Store AI analysis in JSON field only
                article['genai_analysis'] = analysis
                self.api_count += 1
            
            time.sleep(self.delay_between_requests)
            return article
            
        except Exception as e:
            logger.warning(f"GenAI API failed, using fallback: {e}")
            return self._fallback_analysis(article)
    
    def get_analysis_stats(self) -> Dict[str, int]:
        """Get statistics about API vs fallback usage"""
        return {
            'api_count': self.api_count,
            'fallback_count': self.fallback_count,
            'total': self.api_count + self.fallback_count
        }
    
    def _prepare_content(self, article: Dict[str, Any]) -> str:
        """Prepare article content for AI analysis"""
        title = article.get('title', '')
        summary = article.get('summary', '')
        source = article.get('source', '')
        
        content = f"""
Title: {title}
Source: {source}
Summary: {summary}
"""
        return content.strip()
    
    def _create_analysis_schema(self) -> types.Schema:
        """Create structured response schema for analysis"""
        return types.Schema(
            type=types.Type.OBJECT,
            properties={
                "summary_enhanced": types.Schema(type=types.Type.STRING),
                "key_insights": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING)
                ),
                "themes_enhanced": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING)
                ),
                "significance_score": types.Schema(type=types.Type.NUMBER),
                "impact_areas": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING)
                ),
                "technical_complexity": types.Schema(
                    type=types.Type.STRING,
                    enum=["high", "medium", "low"]
                ),
                "business_relevance": types.Schema(
                    type=types.Type.STRING,
                    enum=["high", "medium", "low"]
                ),
                "research_vs_commercial": types.Schema(
                    type=types.Type.STRING,
                    enum=["research", "commercial", "both"]
                ),
                "innovation_level": types.Schema(
                    type=types.Type.STRING,
                    enum=["breakthrough", "incremental", "maintenance"]
                )
            },
            required=["summary_enhanced", "key_insights", "themes_enhanced", 
                     "significance_score", "technical_complexity"]
        )
    
    def _get_genai_analysis_structured(self, content: str) -> Optional[Dict[str, Any]]:
        """Get structured analysis from Google Gemini using new SDK"""
        if self.client is None:
            logger.error("GenAI client is unexpectedly None in _get_genai_analysis_structured.")
            return None
        
        system_instruction = """
You are an expert AI news analyst. Analyze AI news articles and provide structured insights focusing on:
1. Technical significance and innovation level
2. Business and market impact potential  
3. Ethical and societal implications
4. Future trend indicators
5. Key players and competitive landscape

Be precise and analytical in your assessment.
"""
        
        prompt = f"""
Analyze this AI news article:

Title: {content.split('Summary:')[0].replace('Title:', '').strip()}
Content: {content.split('Summary:')[1] if 'Summary:' in content else content}

Provide detailed analysis covering:
- Enhanced summary with key technical details
- 3-5 specific insights about implications
- Main AI/ML themes present
- Significance score (0.0-1.0) based on innovation and impact
- Areas that will be impacted
- Technical complexity level
- Business vs research focus
- Innovation classification
"""
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=self.temperature,
                        response_schema=self._create_analysis_schema(),
                        response_mime_type="application/json"
                    )
                )
                
                if response and response.text:
                    try:
                        analysis = json.loads(response.text)
                        return analysis
                    except json.JSONDecodeError:
                        # Fallback to text parsing if JSON fails
                        return self._parse_fallback_response(response.text)
                        
            except Exception as e:
                logger.warning(f"GenAI request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) + self.delay_between_requests
                    time.sleep(wait_time)
        
        return None
    
    def _parse_fallback_response(self, response_text: str) -> Dict[str, Any]:
        """Parse response when structured JSON fails"""
        try:
            # Try to extract JSON from text
            if '{' in response_text and '}' in response_text:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_part = response_text[start:end]
                return json.loads(json_part)
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"JSON parsing failed: {e}")
            pass
        
        # Return basic structure if parsing fails
        return {
            "summary_enhanced": "Analysis completed with limited structured data",
            "key_insights": ["Advanced AI analysis performed"],
            "themes_enhanced": ["Artificial Intelligence"],
            "significance_score": 0.5,
            "technical_complexity": "medium",
            "business_relevance": "medium",
            "research_vs_commercial": "both",
            "innovation_level": "incremental"
        }
    
    def _fallback_analysis(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when GenAI is not available"""
        self.fallback_count += 1
        
        title = article.get('title', '').lower()
        summary = article.get('summary', '').lower()
        content = f"{title} {summary}"
        
        # Simple keyword-based analysis
        analysis = {
            'summary_enhanced': article.get('summary', 'No summary available'),
            'key_insights': self._extract_fallback_insights(content),
            'themes_enhanced': self._extract_fallback_themes(content),
            'significance_score': self._calculate_fallback_significance(content),
            'impact_areas': self._extract_fallback_impact_areas(content),
            'technical_complexity': self._assess_fallback_complexity(content),
            'business_relevance': self._assess_fallback_business_relevance(content),
            'research_vs_commercial': self._assess_fallback_research_commercial(content),
            'innovation_level': self._assess_fallback_innovation(content)
        }
        
        article['genai_analysis'] = analysis
        
        return article
    
    def _extract_fallback_insights(self, content: str) -> List[str]:
        """Extract insights using keyword matching"""
        insights = []
        
        if any(word in content for word in ['breakthrough', 'revolutionary', 'first']):
            insights.append("Represents a significant technological advancement")
        
        if any(word in content for word in ['market', 'billion', 'funding', 'investment']):
            insights.append("Has significant market and financial implications")
        
        if any(word in content for word in ['ethics', 'regulation', 'policy', 'governance']):
            insights.append("Involves important ethical and regulatory considerations")
        
        if any(word in content for word in ['open source', 'released', 'available']):
            insights.append("May accelerate broader AI development and adoption")
        
        return insights[:3]  # Limit to 3 insights
    
    def _extract_fallback_themes(self, content: str) -> List[str]:
        """Extract themes using keyword matching"""
        themes = []
        
        theme_keywords = {
            'Generative AI': ['gpt', 'chatgpt', 'llm', 'language model', 'generative'],
            'Computer Vision': ['vision', 'image', 'visual', 'opencv', 'cnn'],
            'Machine Learning': ['ml', 'algorithm', 'training', 'model'],
            'AI Ethics': ['ethics', 'bias', 'fairness', 'responsible'],
            'AI Regulation': ['regulation', 'policy', 'governance', 'law'],
            'Deep Learning': ['neural', 'deep learning', 'tensorflow', 'pytorch'],
            'Natural Language Processing': ['nlp', 'language', 'text', 'speech'],
            'AI Applications': ['application', 'implementation', 'deployment']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in content for keyword in keywords):
                themes.append(theme)
        
        return themes[:3]  # Limit to 3 themes
    
    def _calculate_fallback_significance(self, content: str) -> float:
        """Calculate significance score based on keywords"""
        high_impact_words = ['breakthrough', 'revolutionary', 'first', 'new', 'novel']
        medium_impact_words = ['improved', 'enhanced', 'better', 'advanced']
        
        high_count = sum(1 for word in high_impact_words if word in content)
        medium_count = sum(1 for word in medium_impact_words if word in content)
        
        score = min(0.3 + (high_count * 0.2) + (medium_count * 0.1), 1.0)
        return round(score, 2)
    
    def _extract_fallback_impact_areas(self, content: str) -> List[str]:
        """Extract impact areas using keyword matching"""
        areas = []
        
        impact_keywords = {
            'Healthcare': ['health', 'medical', 'diagnosis', 'treatment'],
            'Education': ['education', 'learning', 'teaching', 'student'],
            'Business': ['business', 'enterprise', 'company', 'commercial'],
            'Research': ['research', 'academic', 'university', 'study'],
            'Technology': ['tech', 'software', 'hardware', 'platform'],
            'Society': ['social', 'society', 'public', 'community']
        }
        
        for area, keywords in impact_keywords.items():
            if any(keyword in content for keyword in keywords):
                areas.append(area)
        
        return areas[:2]  # Limit to 2 areas
    
    def _assess_fallback_complexity(self, content: str) -> str:
        """Assess technical complexity"""
        complex_words = ['algorithm', 'neural', 'deep', 'architecture', 'training']
        complex_count = sum(1 for word in complex_words if word in content)
        
        if complex_count >= 3:
            return 'high'
        elif complex_count >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _assess_fallback_business_relevance(self, content: str) -> str:
        """Assess business relevance"""
        business_words = ['market', 'revenue', 'profit', 'business', 'commercial', 'enterprise']
        business_count = sum(1 for word in business_words if word in content)
        
        if business_count >= 2:
            return 'high'
        elif business_count >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _assess_fallback_research_commercial(self, content: str) -> str:
        """Assess research vs commercial nature"""
        research_words = ['research', 'academic', 'university', 'paper', 'study']
        commercial_words = ['product', 'launch', 'release', 'market', 'customer']
        
        research_count = sum(1 for word in research_words if word in content)
        commercial_count = sum(1 for word in commercial_words if word in content)
        
        if research_count > commercial_count:
            return 'research'
        elif commercial_count > research_count:
            return 'commercial'
        else:
            return 'both'
    
    def _assess_fallback_innovation(self, content: str) -> str:
        """Assess innovation level"""
        breakthrough_words = ['breakthrough', 'revolutionary', 'first', 'novel']
        incremental_words = ['improved', 'enhanced', 'better', 'optimized']
        
        if any(word in content for word in breakthrough_words):
            return 'breakthrough'
        elif any(word in content for word in incremental_words):
            return 'incremental'
        else:
            return 'maintenance'

    def analyze_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze multiple articles with improved progress tracking and error handling"""
        from tqdm import tqdm
        
        analyzed_articles = []
        total = len(articles)
        success_count = 0
        
        logger.info(f"🤖 Starting GenAI batch analysis of {total} articles")
        
        with tqdm(total=total, desc="GenAI Analysis", unit="article") as pbar:
            for i, article in enumerate(articles, 1):
                try:
                    analyzed_article = self.analyze_article(article)
                    analyzed_articles.append(analyzed_article)
                    
                    # Check if analysis was successful
                    if 'genai_analysis' in analyzed_article:
                        success_count += 1
                        pbar.set_postfix_str(f"✅ {success_count}/{i} successful")
                    else:
                        pbar.set_postfix_str(f"⚠️ {success_count}/{i} successful")
                        
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning(f"Failed to analyze article {i}: {e}")
                    analyzed_articles.append(article)
                    pbar.set_postfix_str(f"❌ {success_count}/{i} successful")
                
                pbar.update(1)
        
        logger.info(f"✅ GenAI batch analysis completed: {success_count}/{total} successful analyses")
        return analyzed_articles