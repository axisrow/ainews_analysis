#!/usr/bin/env python3
"""
Optimized bulk data collector for AI news
Uses parallel processing to speed up data collection
"""

import sys
import threading
import concurrent.futures
from pathlib import Path
import time
from loguru import logger

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import configuration
from config import get_config, set_bulk_mode

from scrapers.web_scraper import WebScraper
from scrapers.rss_scraper import RSSFeedScraper
from scrapers.reddit_scraper import RedditScraper
from analyzers.nlp_analyzer import NLPAnalyzer
from analyzers.sentiment_analyzer import SentimentAnalyzer
from analyzers.genai_analyzer import GenAIAnalyzer
from models.database import Database


def setup_logging():
    """Setup logging configuration"""
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>"
    )


def collect_rss_data(config):
    """Collect RSS data in parallel"""
    try:
        logger.info("🔄 Starting RSS collection...")
        rss_scraper = RSSFeedScraper(config['data_collection'])
        articles = rss_scraper.scrape_all_feeds()
        logger.info(f"✅ RSS: {len(articles)} articles")
        return articles
    except Exception as e:
        logger.error(f"❌ RSS collection failed: {e}")
        return []


def collect_web_data(config):
    """Collect web scraping data"""
    try:
        logger.info("🔄 Starting web scraping...")
        web_scraper = WebScraper(config['data_collection']['web_scraping'])
        articles = web_scraper.scrape_all_sites()
        logger.info(f"✅ Web: {len(articles)} articles")
        return articles
    except Exception as e:
        logger.error(f"❌ Web scraping failed: {e}")
        return []


def collect_reddit_data(config):
    """Collect Reddit data"""
    try:
        logger.info("🔄 Starting Reddit collection...")
        reddit_scraper = RedditScraper(config['data_collection']['reddit'])
        articles = reddit_scraper.scrape_all_subreddits()
        logger.info(f"✅ Reddit: {len(articles)} articles")
        return articles
    except Exception as e:
        logger.error(f"❌ Reddit collection failed: {e}")
        return []


def parallel_collect_data(config):
    """Collect data from all sources in parallel"""
    all_articles = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_rss = executor.submit(collect_rss_data, config)
        future_web = executor.submit(collect_web_data, config)
        future_reddit = executor.submit(collect_reddit_data, config)
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed([future_rss, future_web, future_reddit]):
            try:
                articles = future.result(timeout=120)  # 2 minute timeout per source
                all_articles.extend(articles)
            except concurrent.futures.TimeoutError:
                logger.warning("⏰ Source timed out, continuing with others...")
            except Exception as e:
                logger.error(f"❌ Source failed: {e}")
    
    # Add keywords to config for each article
    for article in all_articles:
        article['config_keywords'] = config['data_collection']['search_keywords']
    
    return all_articles


def analyze_articles_batch(articles, config):
    """Analyze articles with progress updates"""
    if not articles:
        return articles
    
    logger.info(f"🧠 Analyzing {len(articles)} articles...")
    
    nlp_analyzer = NLPAnalyzer(config['nlp'])
    sentiment_analyzer = SentimentAnalyzer(config['nlp'])
    genai_analyzer = GenAIAnalyzer(config['nlp'])
    
    # Check if GenAI is enabled
    genai_enabled = config['nlp'].get('genai', {}).get('enabled', False)
    
    analyzed_articles = []
    batch_size = 10
    
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        
        for article in batch:
            try:
                # NLP analysis
                article = nlp_analyzer.analyze_article(article)
                # Sentiment analysis
                article = sentiment_analyzer.analyze_article_sentiment(article)
                # GenAI analysis (if enabled)
                if genai_enabled:
                    article = genai_analyzer.analyze_article(article)
                analyzed_articles.append(article)
            except Exception as e:
                logger.warning(f"Analysis failed for article: {e}")
                analyzed_articles.append(article)  # Add without analysis
        
        # Progress update
        progress = min(i + batch_size, len(articles))
        logger.info(f"📊 Progress: {progress}/{len(articles)} articles analyzed")
    
    return analyzed_articles


def bulk_collect(runs=5):
    """Run bulk collection multiple times"""
    setup_logging()
    
    # Load configuration with bulk mode settings
    config = set_bulk_mode()
    
    # Fine-tune bulk settings
    config['data_collection']['reddit']['limit'] = 200
    config['data_collection']['web_scraping']['delay_between_requests'] = 0.3
    config['data_collection']['web_scraping']['max_articles_per_site'] = 50
    
    db = Database(config['database'])
    total_new_articles = 0
    
    logger.info(f"🚀 Starting bulk collection: {runs} runs")
    start_time = time.time()
    
    for run in range(runs):
        run_start = time.time()
        logger.info(f"📦 Run {run + 1}/{runs}")
        
        try:
            # Parallel data collection
            articles = parallel_collect_data(config)
            logger.info(f"📥 Collected {len(articles)} articles")
            
            if articles:
                # Analyze articles
                articles = analyze_articles_batch(articles, config)
                
                # Save to database
                new_count = 0
                for article in articles:
                    if db.save_article(article):
                        new_count += 1
                
                total_new_articles += new_count
                logger.info(f"💾 Saved {new_count} new articles")
            
            run_time = time.time() - run_start
            logger.info(f"⏱️  Run {run + 1} completed in {run_time:.1f}s")
            
            # Short delay between runs
            if run < runs - 1:
                time.sleep(2)
                
        except KeyboardInterrupt:
            logger.info("⏹️  Interrupted by user")
            break
        except Exception as e:
            logger.error(f"❌ Run {run + 1} failed: {e}")
    
    total_time = time.time() - start_time
    logger.info(f"🎉 Bulk collection completed!")
    logger.info(f"📊 Total new articles: {total_new_articles}")
    logger.info(f"⏱️  Total time: {total_time:.1f}s")
    logger.info(f"📈 Rate: {total_new_articles/total_time*60:.1f} articles/minute")
    
    # Generate final report
    current_total = db.get_article_count()
    logger.info(f"🗄️  Database now contains: {current_total} total articles")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Bulk AI News Collector')
    parser.add_argument('--runs', type=int, default=5, help='Number of collection runs')
    
    args = parser.parse_args()
    
    try:
        bulk_collect(args.runs)
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")