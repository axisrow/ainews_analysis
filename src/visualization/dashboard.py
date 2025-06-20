import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from src.models.database import Database
from src.analyzers.keyword_analyzer import KeywordAnalyzer


def run_dashboard(config: dict):
    """Run Streamlit dashboard"""
    st.set_page_config(
        page_title="AI News Research Dashboard",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize database
    db = Database(config['database'])
    
    # Sidebar
    st.sidebar.title("🤖 AI News Research")
    st.sidebar.markdown("---")
    
    # Get database statistics first
    db_stats = db.get_statistics()
    total_articles_in_db = db_stats['total_articles']
    
    # Count articles with dates (simpler approach)
    articles_with_dates_count = len([a for a in db.get_articles(limit=1000) if a.get('published_date')])
    
    # Date filter
    st.sidebar.subheader("Filters")
    st.sidebar.info(f"📊 Total articles: **{total_articles_in_db}**")
    st.sidebar.info(f"📅 Articles with dates: **{articles_with_dates_count}**")
    
    # Use the full available date range from all articles with dates
    default_start = datetime.now() - timedelta(days=365)  # Default to last year
    default_end = datetime.now()
    
    # Get actual date range from all articles with dates
    if db_stats['date_range']['earliest'] and db_stats['date_range']['latest']:
        try:
            all_earliest = datetime.fromisoformat(db_stats['date_range']['earliest'].replace('Z', '+00:00'))
            all_latest = datetime.fromisoformat(db_stats['date_range']['latest'].replace('Z', '+00:00'))
            
            # Use the full range of available data as default
            default_start = all_earliest.date()
            default_end = all_latest.date()
            
            # Show data range info
            days_span = (all_latest - all_earliest).days + 1
            st.sidebar.info(f"📊 Data range: {all_earliest.strftime('%Y-%m-%d')} to {all_latest.strftime('%Y-%m-%d')} ({days_span} days)")
            
        except Exception as e:
            st.sidebar.error(f"Error parsing date range: {e}")
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(default_start, default_end),
        max_value=datetime.now()
    )
    
    
    # Source filter
    sources = db.get_sources()
    selected_sources = st.sidebar.multiselect(
        "Sources",
        options=sources,
        default=sources
    )
    
    # Sentiment filter
    sentiment_filter = st.sidebar.selectbox(
        "Sentiment",
        options=["All", "Positive", "Negative", "Neutral"]
    )
    
    
    # Main content
    st.title("AI News Research Dashboard")
    st.markdown("Real-time analysis of AI news and trends")
    
    # Show simple data summary
    st.info(f"📊 Showing all articles with publication dates from the selected time range")
    
    # Get filtered articles
    articles = db.get_articles(
        limit=1000,
        start_date=datetime.combine(date_range[0], datetime.min.time()) if len(date_range) > 0 else None,
        end_date=datetime.combine(date_range[1], datetime.max.time()) if len(date_range) > 1 else None,
        sentiment=sentiment_filter.lower() if sentiment_filter != "All" else None
    )
    
    # Filter by selected sources
    if selected_sources:
        articles = [a for a in articles if a['source'] in selected_sources]
    
    # Simple filtering: just show all articles with dates
    articles_with_dates = [a for a in articles if a.get('published_date')]
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        filtered_count = len(articles_with_dates)
        delta_msg = f"of {total_articles_in_db} total"
        st.metric("Articles with Dates", filtered_count, delta=delta_msg)
    
    with col2:
        unique_sources = len(set(a['source'] for a in articles_with_dates))
        total_sources = len(db.get_sources())
        delta_sources = f"of {total_sources} total"
        st.metric("Active Sources", unique_sources, delta=delta_sources)
    
    with col3:
        avg_sentiment = calculate_average_sentiment(articles_with_dates)
        st.metric("Avg Sentiment", f"{avg_sentiment:.2f}", 
                 delta=f"{'Positive' if avg_sentiment > 0 else 'Negative'}")
    
    with col4:
        today_articles = len([a for a in articles_with_dates 
                            if a.get('published_date') and 
                            a['published_date'][:10] == datetime.now().strftime('%Y-%m-%d')])
        st.metric("Today's Articles", today_articles)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["📊 Overview", "😊 Sentiment Analysis", "🏷️ Topics & Entities", 
         "📈 Trends", "🤖 GenAI Insights", "📰 Articles", "🔑 Keywords"]
    )
    
    with tab1:
        show_overview(articles_with_dates, db)
    
    with tab2:
        show_sentiment_analysis(articles_with_dates)
    
    with tab3:
        show_topics_entities(articles_with_dates)
    
    with tab4:
        show_trends(articles_with_dates)
    
    with tab5:
        show_genai_insights(articles_with_dates)
    
    with tab6:
        show_articles(articles_with_dates)
    
    with tab7:
        show_keywords_editor(config)


def calculate_average_sentiment(articles):
    """Calculate average sentiment score"""
    scores = []
    for article in articles:
        if 'sentiment' in article and 'polarity' in article['sentiment']:
            scores.append(article['sentiment']['polarity'])
        elif 'sentiment' in article and article['sentiment']['sentiment'] == 'positive':
            scores.append(0.5)
        elif 'sentiment' in article and article['sentiment']['sentiment'] == 'negative':
            scores.append(-0.5)
        else:
            scores.append(0)
    
    return sum(scores) / len(scores) if scores else 0


def show_overview(articles, db):
    """Show overview tab"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Articles by source
        st.subheader("Articles by Source")
        source_counts = pd.DataFrame(
            [(a['source'], 1) for a in articles],
            columns=['Source', 'Count']
        ).groupby('Source').sum().reset_index()
        
        fig = px.bar(source_counts, x='Source', y='Count',
                    title="Article Distribution by Source")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Articles by type
        st.subheader("Articles by Type")
        type_counts = pd.DataFrame(
            [(a.get('source_type', 'unknown'), 1) for a in articles],
            columns=['Type', 'Count']
        ).groupby('Type').sum().reset_index()
        
        fig = px.pie(type_counts, values='Count', names='Type',
                    title="Distribution by Source Type")
        st.plotly_chart(fig, use_container_width=True)
    
    # Timeline
    st.subheader("Articles Timeline")
    timeline_data = prepare_timeline_data(articles)
    if not timeline_data.empty:
        fig = px.line(timeline_data, x='Date', y='Count',
                     title="Articles Published Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    # Analysis Method Statistics
    st.subheader("Analysis Coverage")
    col1, col2, col3 = st.columns(3)
    
    sentiment_stats = prepare_sentiment_method_stats(articles)
    genai_articles = len([a for a in articles if a.get('genai_analysis')])
    nlp_articles = len([a for a in articles if a.get('themes') or a.get('entities')])
    
    with col1:
        st.metric(
            "😊 Sentiment Analysis",
            sentiment_stats.get('transformer', 0) + sentiment_stats.get('textblob', 0),
            delta=f"T:{sentiment_stats.get('transformer', 0)} B:{sentiment_stats.get('textblob', 0)}"
        )
    
    with col2:
        st.metric(
            "🤖 GenAI Analysis",
            genai_articles,
            delta=f"{genai_articles/len(articles)*100:.1f}% coverage" if articles else "0% coverage"
        )
    
    with col3:
        st.metric(
            "🏷️ NLP Analysis",
            nlp_articles,
            delta=f"{nlp_articles/len(articles)*100:.1f}% coverage" if articles else "0% coverage"
        )


def show_sentiment_analysis(articles):
    """Show sentiment analysis tab"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Overall sentiment distribution
        st.subheader("Overall Sentiment Distribution")
        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        
        for article in articles:
            if 'sentiment' in article:
                sentiment = article['sentiment'].get('sentiment', 'neutral')
                sentiment_counts[sentiment.capitalize()] += 1
        
        fig = px.pie(
            values=list(sentiment_counts.values()),
            names=list(sentiment_counts.keys()),
            title="Sentiment Distribution",
            color_discrete_map={
                'Positive': '#2E7D32',
                'Negative': '#C62828',
                'Neutral': '#757575'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment by source
        st.subheader("Sentiment by Source")
        source_sentiment = prepare_source_sentiment_data(articles)
        
        if not source_sentiment.empty:
            fig = px.bar(
                source_sentiment,
                x='Source',
                y=['Positive', 'Negative', 'Neutral'],
                title="Sentiment Distribution by Source",
                color_discrete_map={
                    'Positive': '#2E7D32',
                    'Negative': '#C62828',
                    'Neutral': '#757575'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment timeline
    st.subheader("Sentiment Over Time")
    sentiment_timeline = prepare_sentiment_timeline(articles)
    
    if not sentiment_timeline.empty:
        fig = px.line(
            sentiment_timeline,
            x='Date',
            y='Sentiment Score',
            title="Average Sentiment Score Over Time",
            line_shape='spline'
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    
    # Aspect-based sentiment analysis
    st.subheader("Aspect-Based Sentiment Analysis")
    aspect_sentiment_data = prepare_aspect_sentiment_data(articles)
    
    if aspect_sentiment_data:
        col1, col2 = st.columns(2)
        
        with col1:
            # Aspect sentiment distribution
            aspect_df = pd.DataFrame([
                {'Aspect': aspect, 'Sentiment': sentiment, 'Score': data['confidence']}
                for aspect, sentiments in aspect_sentiment_data.items()
                for sentiment, data in sentiments.items()
                if data['count'] > 0
            ])
            
            if not aspect_df.empty:
                fig = px.bar(
                    aspect_df,
                    x='Aspect',
                    y='Score',
                    color='Sentiment',
                    title="Sentiment by AI Aspects",
                    color_discrete_map={
                        'positive': '#2E7D32',
                        'negative': '#C62828',
                        'neutral': '#757575'
                    }
                )
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Aspect coverage
            aspect_coverage = {aspect: sum(s['count'] for s in sentiments.values()) 
                             for aspect, sentiments in aspect_sentiment_data.items()}
            
            if aspect_coverage:
                coverage_df = pd.DataFrame(
                    list(aspect_coverage.items()),
                    columns=['Aspect', 'Mentions']
                ).sort_values('Mentions', ascending=False)
                
                fig = px.pie(
                    coverage_df,
                    values='Mentions',
                    names='Aspect',
                    title="AI Aspect Coverage Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment analysis method breakdown
    st.subheader("Sentiment Analysis Methods")
    method_stats = prepare_sentiment_method_stats(articles)
    
    if method_stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🤖 Transformer", method_stats.get('transformer', 0))
        with col2:
            st.metric("📝 TextBlob", method_stats.get('textblob', 0))
        with col3:
            st.metric("❓ Unknown", method_stats.get('unknown', 0))


def show_topics_entities(articles):
    """Show topics and entities tab"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Word cloud
        st.subheader("Word Cloud")
        text = ' '.join([f"{a.get('title', '')} {a.get('summary', '')}" 
                        for a in articles])
        
        if text.strip():
            wordcloud = WordCloud(
                width=800, 
                height=400,
                background_color='white',
                colormap='viridis'
            ).generate(text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    
    with col2:
        # Top themes
        st.subheader("Top Themes")
        theme_counts = {}
        for article in articles:
            for theme in article.get('themes', []):
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        if theme_counts:
            theme_df = pd.DataFrame(
                list(theme_counts.items()),
                columns=['Theme', 'Count']
            ).sort_values('Count', ascending=False).head(10)
            
            fig = px.bar(
                theme_df,
                x='Count',
                y='Theme',
                orientation='h',
                title="Top AI Themes"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Topic Modeling Results
    st.subheader("Topic Modeling Results")
    topic_modeling_data = prepare_topic_modeling_data(articles)
    
    if topic_modeling_data:
        col1, col2 = st.columns(2)
        
        with col1:
            # Display discovered topics
            st.markdown("**Discovered Topics:**")
            for i, topic in enumerate(topic_modeling_data[:5]):  # Show top 5 topics
                st.write(f"**Topic {i+1}:** {', '.join(topic.get('words', []))}")
                if topic.get('weight'):
                    st.progress(min(topic['weight'], 1.0))
        
        with col2:
            # Topic weights visualization
            if len(topic_modeling_data) > 1:
                topic_df = pd.DataFrame([
                    {'Topic': f"Topic {t['topic_id']+1}", 'Weight': t.get('weight', 0)}
                    for t in topic_modeling_data
                ])
                
                fig = px.bar(
                    topic_df,
                    x='Topic',
                    y='Weight',
                    title="Topic Model Weights"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Key Phrases Analysis
    st.subheader("Key Phrases Analysis")
    key_phrases_data = prepare_key_phrases_data(articles)
    
    if key_phrases_data:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top key phrases
            phrases_df = pd.DataFrame(
                list(key_phrases_data.items()),
                columns=['Phrase', 'Frequency']
            ).sort_values('Frequency', ascending=False).head(15)
            
            fig = px.bar(
                phrases_df,
                x='Frequency',
                y='Phrase',
                orientation='h',
                title="Most Common Key Phrases"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Key phrases word cloud
            if key_phrases_data:
                phrases_text = ' '.join([phrase for phrase in key_phrases_data.keys()])
                
                wordcloud = WordCloud(
                    width=400,
                    height=300,
                    background_color='white',
                    colormap='plasma'
                ).generate(phrases_text)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title("Key Phrases Word Cloud")
                st.pyplot(fig)
    
    # Top entities
    st.subheader("Top Entities")
    entity_types = ['ORG', 'PERSON', 'PRODUCT']
    
    cols = st.columns(len(entity_types))
    for i, entity_type in enumerate(entity_types):
        with cols[i]:
            entities = extract_top_entities(articles, entity_type, top_n=10)
            if entities:
                st.markdown(f"**{entity_type}**")
                for entity, count in entities:
                    st.write(f"• {entity} ({count})")


def show_trends(articles):
    """Show trends tab"""
    st.subheader("Theme Trends Over Time")
    
    # Select themes to track
    all_themes = set()
    for article in articles:
        all_themes.update(article.get('themes', []))
    
    selected_themes = st.multiselect(
        "Select themes to track",
        options=sorted(all_themes),
        default=list(sorted(all_themes))[:5] if all_themes else []
    )
    
    if selected_themes:
        theme_timeline = prepare_theme_timeline(articles, selected_themes)
        
        if not theme_timeline.empty:
            fig = px.line(
                theme_timeline,
                x='Date',
                y='Count',
                color='Theme',
                title="Theme Mentions Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Reddit engagement trends
    reddit_articles = [a for a in articles if a.get('source_type') == 'reddit']
    if reddit_articles:
        st.subheader("Reddit Engagement Trends")
        
        engagement_data = prepare_reddit_engagement_data(reddit_articles)
        if not engagement_data.empty:
            fig = px.scatter(
                engagement_data,
                x='Published Date',
                y='Score',
                size='Comments',
                color='Sentiment',
                hover_data=['Title'],
                title="Reddit Post Engagement"
            )
            st.plotly_chart(fig, use_container_width=True)


def show_genai_insights(articles):
    """Show GenAI analysis insights tab"""
    st.subheader("🤖 Google GenAI Analysis")
    
    # Filter articles with GenAI analysis
    genai_articles = [a for a in articles if a is not None and 'genai_analysis' in a and a.get('genai_analysis') is not None]
    
    if not genai_articles:
        st.info("📝 No GenAI analysis available yet. To enable:")
        st.code("""
# 1. Get Google AI API key from https://aistudio.google.com/app/apikey
# 2. Add to .env file:
GOOGLE_API_KEY=your_api_key_here

# 3. Install updated dependency:
pip install google-genai[aiohttp]

# 4. Re-run analysis:
python main.py --analyze
        """)
        
        st.markdown("### 🆕 New GenAI Features:")
        st.markdown("""
        - **Enhanced Model**: Now using Gemini 2.0 Flash for better accuracy
        - **Structured Responses**: Reliable JSON parsing with schema validation  
        - **Safety Filtering**: Content safety controls and filtering
        - **Async Support**: Optional async processing for better performance
        - **Better Error Handling**: Improved retry logic and fallback analysis
        """)
        return
    
    # Show analysis method breakdown
    v1_count = sum(1 for a in genai_articles if a.get('analysis_method') != 'genai_v2')
    v2_count = sum(1 for a in genai_articles if a.get('analysis_method') == 'genai_v2')
    
    col_status1, col_status2, col_status3 = st.columns(3)
    with col_status1:
        st.success(f"✅ {len(genai_articles)} articles analyzed")
    with col_status2:
        if v2_count > 0:
            st.info(f"🆕 {v2_count} with new SDK")
    with col_status3:
        if v1_count > 0:
            st.warning(f"⚠️ {v1_count} with legacy analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Significance scores distribution
        st.subheader("📈 Significance Scores")
        scores = []
        for article in genai_articles:
            genai_analysis = article.get('genai_analysis')
            if genai_analysis and isinstance(genai_analysis, dict):
                score = genai_analysis.get('significance_score', 0)
                if score and isinstance(score, (int, float)):
                    scores.append(score)
        
        if scores:
            fig = px.histogram(
                scores, 
                nbins=10,
                title="Distribution of Article Significance Scores",
                labels={'x': 'Significance Score', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Technical complexity distribution
        st.subheader("🔧 Technical Complexity")
        complexity_counts = {'High': 0, 'Medium': 0, 'Low': 0}
        
        for article in genai_articles:
            genai_analysis = article.get('genai_analysis')
            if genai_analysis and isinstance(genai_analysis, dict):
                complexity = genai_analysis.get('technical_complexity', 'unknown')
                if isinstance(complexity, str) and complexity.capitalize() in complexity_counts:
                    complexity_counts[complexity.capitalize()] += 1
        
        if any(complexity_counts.values()):
            fig = px.pie(
                values=list(complexity_counts.values()),
                names=list(complexity_counts.keys()),
                title="Technical Complexity Distribution",
                color_discrete_map={
                    'High': '#FF6B6B',
                    'Medium': '#4ECDC4', 
                    'Low': '#45B7D1'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Innovation level analysis
    st.subheader("💡 Innovation Level Analysis")
    col1, col2, col3 = st.columns(3)
    
    innovation_counts = {'Breakthrough': 0, 'Incremental': 0, 'Maintenance': 0}
    for article in genai_articles:
        genai_analysis = article.get('genai_analysis')
        if genai_analysis and isinstance(genai_analysis, dict):
            innovation = genai_analysis.get('innovation_level', 'unknown')
            if isinstance(innovation, str) and innovation.capitalize() in innovation_counts:
                innovation_counts[innovation.capitalize()] += 1
    
    with col1:
        st.metric("🚀 Breakthrough", innovation_counts['Breakthrough'])
    with col2:
        st.metric("📈 Incremental", innovation_counts['Incremental'])
    with col3:
        st.metric("🔧 Maintenance", innovation_counts['Maintenance'])
    
    # Top insights
    st.subheader("🎯 Key Insights")
    all_insights = []
    for article in genai_articles:
        genai_analysis = article.get('genai_analysis')
        if genai_analysis and isinstance(genai_analysis, dict):
            insights = genai_analysis.get('key_insights', [])
            if isinstance(insights, list):
                all_insights.extend(insights)
    
    if all_insights:
        insight_counts = {}
        for insight in all_insights:
            insight_counts[insight] = insight_counts.get(insight, 0) + 1
        
        # Show top insights
        top_insights = sorted(insight_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for insight, count in top_insights:
            st.write(f"• **{insight}** (mentioned {count} times)")
    
    # Impact areas analysis
    st.subheader("🌍 Impact Areas")
    all_impact_areas = []
    for article in genai_articles:
        genai_analysis = article.get('genai_analysis')
        if genai_analysis and isinstance(genai_analysis, dict):
            areas = genai_analysis.get('impact_areas', [])
            if isinstance(areas, list):
                all_impact_areas.extend(areas)
    
    if all_impact_areas:
        area_counts = {}
        for area in all_impact_areas:
            area_counts[area] = area_counts.get(area, 0) + 1
        
        if area_counts:
            area_df = pd.DataFrame(
                list(area_counts.items()),
                columns=['Impact Area', 'Count']
            ).sort_values('Count', ascending=False)
            
            fig = px.bar(
                area_df,
                x='Count',
                y='Impact Area',
                orientation='h',
                title="Most Impacted Areas by AI News"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Research vs Commercial analysis
    st.subheader("🔬 Research vs Commercial Focus")
    col1, col2, col3 = st.columns(3)
    
    focus_counts = {'Research': 0, 'Commercial': 0, 'Both': 0}
    for article in genai_articles:
        genai_analysis = article.get('genai_analysis')
        if genai_analysis and isinstance(genai_analysis, dict):
            focus = genai_analysis.get('research_vs_commercial', 'unknown')
            if isinstance(focus, str) and focus.capitalize() in focus_counts:
                focus_counts[focus.capitalize()] += 1
    
    with col1:
        st.metric("🔬 Research", focus_counts['Research'])
    with col2:
        st.metric("💼 Commercial", focus_counts['Commercial'])
    with col3:
        st.metric("🤝 Both", focus_counts['Both'])
    
    # Performance metrics for new SDK
    if v2_count > 0:
        st.subheader("🚀 New SDK Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate average scores for v2 articles
        v2_articles = [a for a in genai_articles if a.get('analysis_method') == 'genai_v2']
        valid_scores = []
        for a in v2_articles:
            genai_analysis = a.get('genai_analysis')
            if genai_analysis and isinstance(genai_analysis, dict):
                score = genai_analysis.get('significance_score', 0)
                if isinstance(score, (int, float)):
                    valid_scores.append(score)
        avg_significance = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        
        with col1:
            st.metric("📊 Avg Significance", f"{avg_significance:.2f}")
        with col2:
            high_impact = 0
            for a in v2_articles:
                genai_analysis = a.get('genai_analysis')
                if genai_analysis and isinstance(genai_analysis, dict):
                    score = genai_analysis.get('significance_score', 0)
                    if isinstance(score, (int, float)) and score > 0.7:
                        high_impact += 1
            st.metric("🔥 High Impact", high_impact)
        with col3:
            breakthrough_count = 0
            for a in v2_articles:
                genai_analysis = a.get('genai_analysis')
                if genai_analysis and isinstance(genai_analysis, dict):
                    innovation = genai_analysis.get('innovation_level')
                    if isinstance(innovation, str) and innovation.lower() == 'breakthrough':
                        breakthrough_count += 1
            st.metric("💥 Breakthroughs", breakthrough_count)
        with col4:
            high_complexity = 0
            for a in v2_articles:
                genai_analysis = a.get('genai_analysis')
                if genai_analysis and isinstance(genai_analysis, dict):
                    complexity = genai_analysis.get('technical_complexity')
                    if isinstance(complexity, str) and complexity.lower() == 'high':
                        high_complexity += 1
            st.metric("🧠 High Complexity", high_complexity)


def show_articles(articles):
    """Show articles tab"""
    st.subheader("Article Browser")
    
    # Search
    search_term = st.text_input("Search articles", "")
    
    # Filter articles
    filtered_articles = articles
    if search_term:
        filtered_articles = [
            a for a in articles
            if search_term.lower() in a.get('title', '').lower() or
               search_term.lower() in a.get('summary', '').lower()
        ]
    
    st.write(f"Showing {len(filtered_articles)} articles")
    
    # Display articles
    for article in filtered_articles[:50]:  # Limit display
        with st.expander(f"{article.get('title', 'Untitled')} - {article.get('source', 'Unknown')}"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Published:** {article.get('published_date', 'Unknown')[:10]}")
                if article.get('author'):
                    st.write(f"**Author:** {article['author']}")
            
            with col2:
                sentiment = article.get('sentiment', {}).get('sentiment', 'unknown')
                sentiment_color = {
                    'positive': '🟢',
                    'negative': '🔴',
                    'neutral': '⚪'
                }.get(sentiment, '⚫')
                st.write(f"**Sentiment:** {sentiment_color} {sentiment}")
            
            with col3:
                if article.get('reddit_score'):
                    st.write(f"**Score:** {article['reddit_score']}")
                    st.write(f"**Comments:** {article.get('reddit_comments', 0)}")
            
            st.write("**Summary:**")
            st.write(article.get('summary', 'No summary available'))
            
            if article.get('themes'):
                st.write(f"**Themes:** {', '.join(article['themes'])}")
            
            # Show GenAI insights if available
            if article.get('genai_analysis'):
                genai = article['genai_analysis']
                
                if genai.get('significance_score'):
                    st.write(f"**AI Significance:** {genai['significance_score']:.2f}/1.0")
                
                if genai.get('key_insights'):
                    st.write("**AI Insights:**")
                    for insight in genai['key_insights'][:2]:  # Show first 2 insights
                        st.write(f"  • {insight}")
                
                if genai.get('innovation_level'):
                    level_emoji = {'breakthrough': '🚀', 'incremental': '📈', 'maintenance': '🔧'}
                    emoji = level_emoji.get(genai['innovation_level'], '❓')
                    st.write(f"**Innovation Level:** {emoji} {genai['innovation_level'].capitalize()}")
            
            if article.get('url'):
                st.write(f"[Read more]({article['url']})")


# Helper functions
def prepare_timeline_data(articles):
    """Prepare timeline data for visualization - show all articles with dates"""
    timeline = {}
    for article in articles:
        # Show all articles with valid publication dates
        if article.get('published_date'):
            date = article['published_date'][:10]
            if date:
                timeline[date] = timeline.get(date, 0) + 1
    
    if timeline:
        df = pd.DataFrame(list(timeline.items()), columns=['Date', 'Count'])
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values('Date')
    return pd.DataFrame()


def prepare_source_sentiment_data(articles):
    """Prepare sentiment by source data"""
    data = {}
    for article in articles:
        source = article.get('source', 'Unknown')
        sentiment = article.get('sentiment', {}).get('sentiment', 'neutral')
        
        if source not in data:
            data[source] = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        
        data[source][sentiment.capitalize()] += 1
    
    if data:
        df = pd.DataFrame.from_dict(data, orient='index').reset_index()
        df.columns = ['Source', 'Positive', 'Negative', 'Neutral']
        return df
    return pd.DataFrame()


def prepare_sentiment_timeline(articles):
    """Prepare sentiment timeline data"""
    timeline = {}
    for article in articles:
        date = article.get('published_date', '')[:10]
        if date and 'sentiment' in article:
            if date not in timeline:
                timeline[date] = []
            
            # Get sentiment score
            if 'polarity' in article['sentiment']:
                score = article['sentiment']['polarity']
            elif article['sentiment']['sentiment'] == 'positive':
                score = 0.5
            elif article['sentiment']['sentiment'] == 'negative':
                score = -0.5
            else:
                score = 0
            
            timeline[date].append(score)
    
    # Calculate average per day
    avg_timeline = []
    for date, scores in timeline.items():
        avg_timeline.append({
            'Date': date,
            'Sentiment Score': sum(scores) / len(scores)
        })
    
    if avg_timeline:
        df = pd.DataFrame(avg_timeline)
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values('Date')
    return pd.DataFrame()


def extract_top_entities(articles, entity_type, top_n=10):
    """Extract top entities of a specific type"""
    entity_counts = {}
    for article in articles:
        entities = article.get('entities', {}).get(entity_type, [])
        for entity in entities:
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
    
    return sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]


def prepare_theme_timeline(articles, selected_themes):
    """Prepare theme timeline data"""
    timeline_data = []
    
    # Group by date
    date_themes = {}
    for article in articles:
        date = article.get('published_date', '')[:10]
        if date:
            if date not in date_themes:
                date_themes[date] = {theme: 0 for theme in selected_themes}
            
            for theme in article.get('themes', []):
                if theme in selected_themes:
                    date_themes[date][theme] += 1
    
    # Convert to dataframe format
    for date, themes in date_themes.items():
        for theme, count in themes.items():
            timeline_data.append({
                'Date': date,
                'Theme': theme,
                'Count': count
            })
    
    if timeline_data:
        df = pd.DataFrame(timeline_data)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return pd.DataFrame()


def prepare_reddit_engagement_data(reddit_articles):
    """Prepare Reddit engagement data"""
    data = []
    for article in reddit_articles:
        if article.get('reddit_score') is not None:
            data.append({
                'Title': article.get('title', '')[:50] + '...',
                'Published Date': article.get('published_date', '')[:10],
                'Score': article['reddit_score'],
                'Comments': article.get('reddit_comments', 0),
                'Sentiment': article.get('sentiment', {}).get('sentiment', 'neutral').capitalize()
            })
    
    if data:
        df = pd.DataFrame(data)
        df['Published Date'] = pd.to_datetime(df['Published Date'])
        return df
    return pd.DataFrame()


def prepare_aspect_sentiment_data(articles):
    """Prepare aspect-based sentiment analysis data"""
    aspect_data = {}
    
    for article in articles:
        sentiment_aspects = article.get('sentiment_aspects', {})
        
        if sentiment_aspects and isinstance(sentiment_aspects, dict):
            for aspect, aspect_sentiment in sentiment_aspects.items():
                if aspect not in aspect_data:
                    aspect_data[aspect] = {
                        'positive': {'count': 0, 'confidence': 0},
                        'negative': {'count': 0, 'confidence': 0},
                        'neutral': {'count': 0, 'confidence': 0}
                    }
                
                if isinstance(aspect_sentiment, dict):
                    sentiment = aspect_sentiment.get('sentiment', 'neutral')
                    confidence = aspect_sentiment.get('confidence', 0)
                    
                    if sentiment in aspect_data[aspect]:
                        aspect_data[aspect][sentiment]['count'] += 1
                        aspect_data[aspect][sentiment]['confidence'] += confidence
    
    # Calculate average confidence
    for aspect in aspect_data:
        for sentiment in aspect_data[aspect]:
            count = aspect_data[aspect][sentiment]['count']
            if count > 0:
                aspect_data[aspect][sentiment]['confidence'] /= count
    
    return aspect_data


def prepare_sentiment_method_stats(articles):
    """Prepare sentiment analysis method statistics"""
    stats = {'transformer': 0, 'textblob': 0, 'unknown': 0}
    
    for article in articles:
        sentiment_data = article.get('sentiment', {})
        if isinstance(sentiment_data, dict):
            method = sentiment_data.get('method', 'unknown')
            if method in stats:
                stats[method] += 1
            else:
                stats['unknown'] += 1
    
    return stats


def prepare_topic_modeling_data(articles):
    """Prepare topic modeling data from corpus analysis"""
    # Look for topic modeling results in the first article that has them
    # (since topic modeling is typically done on the entire corpus)
    
    for article in articles:
        if 'topics' in article:
            topics = article['topics']
            if isinstance(topics, list) and topics:
                return topics
    
    # If no stored topic results, try to extract from themes
    # This is a simplified fallback
    all_themes = []
    for article in articles:
        themes = article.get('themes', [])
        if isinstance(themes, list):
            all_themes.extend(themes)
    
    if all_themes:
        from collections import Counter
        theme_counts = Counter(all_themes)
        
        # Convert to topic-like structure
        topics = []
        for i, (theme, count) in enumerate(theme_counts.most_common(10)):
            topics.append({
                'topic_id': i,
                'words': theme.split()[:5],  # Split theme into words
                'weight': count / len(articles)  # Normalize by article count
            })
        
        return topics
    
    return []


def prepare_key_phrases_data(articles):
    """Prepare key phrases data"""
    phrase_counts = {}
    
    for article in articles:
        key_phrases = article.get('key_phrases', [])
        if isinstance(key_phrases, list):
            for phrase in key_phrases:
                if isinstance(phrase, str) and len(phrase.strip()) > 2:
                    phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
    
    # Filter out phrases that appear only once
    filtered_phrases = {phrase: count for phrase, count in phrase_counts.items() 
                       if count > 1}
    
    return filtered_phrases


def show_keywords_editor(config: dict):
    """Show keywords editor tab"""
    st.subheader("🔑 Keywords Editor")
    
    # Initialize keyword analyzer
    keyword_analyzer = KeywordAnalyzer(config)
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Actions")
        
        # Analyze button
        if st.button("🔍 Analyze Keywords", help="Extract keywords from database"):
            with st.spinner("Analyzing keywords from database..."):
                try:
                    db = Database(config['database'])
                    results = keyword_analyzer.analyze_keywords_from_database(
                        database=db,
                        min_frequency=3,
                        top_n=200
                    )
                    
                    if results['status'] == 'success':
                        # Save keywords
                        output_path = keyword_analyzer.save_keywords(results)
                        st.success(f"✅ Keywords analyzed and saved! Found {len(results['keywords'])} keywords.")
                        st.rerun()  # Refresh the page
                    else:
                        st.error(f"❌ Analysis failed: {results.get('message', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
        
        # Load keywords button
        if st.button("📂 Reload Keywords", help="Reload keywords from file"):
            st.rerun()  # Just refresh the page
        
        st.markdown("---")
        
        # Statistics
        st.subheader("📊 Statistics")
        try:
            keywords_data = keyword_analyzer.load_keywords()
            if keywords_data.get('keywords'):
                total_keywords = len(keywords_data['keywords'])
                ai_keywords = len([k for k in keywords_data['keywords'] if k.get('is_ai_term')])
                
                st.metric("Total Keywords", total_keywords)
                st.metric("AI-related", ai_keywords)
                st.metric("General", total_keywords - ai_keywords)
                
                if keywords_data.get('statistics'):
                    stats = keywords_data['statistics']
                    st.metric("Articles Analyzed", stats.get('total_articles', 0))
                    
                    if stats.get('date_range'):
                        date_range = stats['date_range']
                        st.write(f"**Date Range:**")
                        st.write(f"{date_range.get('earliest', 'N/A')[:10]} to {date_range.get('latest', 'N/A')[:10]}")
            else:
                st.info("No keywords loaded. Click 'Analyze Keywords' to generate.")
                
        except Exception as e:
            st.error(f"Error loading statistics: {str(e)}")
    
    with col1:
        st.subheader("Keywords List")
        
        try:
            # Load keywords
            keywords_data = keyword_analyzer.load_keywords()
            
            if not keywords_data.get('keywords'):
                st.info("🔍 No keywords found. Click 'Analyze Keywords' to generate keywords from your articles database.")
                return
            
            keywords = keywords_data['keywords']
            
            # Search and filter controls
            col_search, col_filter = st.columns([2, 1])
            
            with col_search:
                search_term = st.text_input("🔍 Search keywords", placeholder="Type to search...")
            
            with col_filter:
                filter_type = st.selectbox("Filter", ["All", "AI-related only", "General only"])
            
            # Filter keywords based on search and type
            filtered_keywords = keywords
            
            if search_term:
                filtered_keywords = [k for k in filtered_keywords 
                                   if search_term.lower() in k['keyword'].lower()]
            
            if filter_type == "AI-related only":
                filtered_keywords = [k for k in filtered_keywords if k.get('is_ai_term')]
            elif filter_type == "General only":
                filtered_keywords = [k for k in filtered_keywords if not k.get('is_ai_term')]
            
            st.write(f"Showing {len(filtered_keywords)} of {len(keywords)} keywords")
            
            # Keywords editor
            if filtered_keywords:
                # Show keywords in an editable format
                st.markdown("**Edit keywords below:** (✏️ = editable, 🗑️ = delete)")
                
                # Track changes
                if 'keywords_changes' not in st.session_state:
                    st.session_state.keywords_changes = {}
                if 'keywords_to_delete' not in st.session_state:
                    st.session_state.keywords_to_delete = set()
                
                # Display keywords with edit/delete options
                for i, keyword in enumerate(filtered_keywords[:50]):  # Limit to 50 for performance
                    col_kw, col_score, col_ai, col_actions = st.columns([3, 1, 1, 1])
                    
                    with col_kw:
                        key = f"keyword_{i}_{keyword['keyword']}"
                        new_keyword = st.text_input(
                            "Keyword",
                            value=keyword['keyword'],
                            key=key,
                            label_visibility="collapsed"
                        )
                        
                        # Track changes
                        if new_keyword != keyword['keyword']:
                            st.session_state.keywords_changes[keyword['keyword']] = new_keyword
                    
                    with col_score:
                        st.text(f"Score: {keyword['score']:.3f}")
                    
                    with col_ai:
                        ai_emoji = "🤖" if keyword.get('is_ai_term') else "⚪"
                        st.text(ai_emoji)
                    
                    with col_actions:
                        delete_key = f"delete_{i}_{keyword['keyword']}"
                        if st.button("🗑️", key=delete_key, help="Delete keyword"):
                            st.session_state.keywords_to_delete.add(keyword['keyword'])
                            st.rerun()
                
                # Add new keyword section
                st.markdown("---")
                st.subheader("➕ Add New Keyword")
                
                col_new_kw, col_new_ai, col_add = st.columns([2, 1, 1])
                
                with col_new_kw:
                    new_keyword = st.text_input("New keyword", placeholder="Enter new keyword...")
                
                with col_new_ai:
                    is_ai_term = st.checkbox("AI-related", key="new_keyword_ai")
                
                with col_add:
                    if st.button("➕ Add") and new_keyword.strip():
                        # Add new keyword to the list
                        if 'new_keywords' not in st.session_state:
                            st.session_state.new_keywords = []
                        
                        st.session_state.new_keywords.append({
                            'keyword': new_keyword.strip().lower(),
                            'score': 1.0,
                            'sources': ['manual'],
                            'frequency': 1,
                            'is_ai_term': is_ai_term
                        })
                        st.success(f"Added: {new_keyword}")
                        st.rerun()
                
                # Save changes section
                if (st.session_state.get('keywords_changes') or 
                    st.session_state.get('keywords_to_delete') or 
                    st.session_state.get('new_keywords')):
                    
                    st.markdown("---")
                    st.subheader("💾 Save Changes")
                    
                    # Show pending changes
                    if st.session_state.get('keywords_changes'):
                        st.write("**Modified keywords:**")
                        for old, new in st.session_state.keywords_changes.items():
                            st.write(f"• {old} → {new}")
                    
                    if st.session_state.get('keywords_to_delete'):
                        st.write("**Keywords to delete:**")
                        for kw in st.session_state.keywords_to_delete:
                            st.write(f"• {kw}")
                    
                    if st.session_state.get('new_keywords'):
                        st.write("**New keywords:**")
                        for kw in st.session_state.new_keywords:
                            ai_mark = " 🤖" if kw['is_ai_term'] else ""
                            st.write(f"• {kw['keyword']}{ai_mark}")
                    
                    col_save, col_cancel = st.columns(2)
                    
                    with col_save:
                        if st.button("💾 Save All Changes", type="primary"):
                            try:
                                # Apply changes to keywords data
                                updated_keywords = []
                                
                                for keyword in keywords:
                                    # Skip deleted keywords
                                    if keyword['keyword'] in st.session_state.get('keywords_to_delete', set()):
                                        continue
                                    
                                    # Apply modifications
                                    if keyword['keyword'] in st.session_state.get('keywords_changes', {}):
                                        keyword['keyword'] = st.session_state.keywords_changes[keyword['keyword']]
                                    
                                    updated_keywords.append(keyword)
                                
                                # Add new keywords
                                if st.session_state.get('new_keywords'):
                                    updated_keywords.extend(st.session_state.new_keywords)
                                
                                # Update keywords data
                                keywords_data['keywords'] = updated_keywords
                                
                                # Save to file
                                output_path = keyword_analyzer.save_keywords(keywords_data)
                                
                                # Clear session state
                                st.session_state.keywords_changes = {}
                                st.session_state.keywords_to_delete = set()
                                st.session_state.new_keywords = []
                                
                                st.success(f"✅ Changes saved successfully!")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error saving changes: {str(e)}")
                    
                    with col_cancel:
                        if st.button("❌ Cancel Changes"):
                            # Clear session state
                            st.session_state.keywords_changes = {}
                            st.session_state.keywords_to_delete = set()
                            st.session_state.new_keywords = []
                            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error loading keywords: {str(e)}")
            st.write("Try clicking 'Analyze Keywords' to generate new keywords.")


if __name__ == "__main__":
    # For standalone testing
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import get_config
    config = get_config()
    run_dashboard(config)