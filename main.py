#!/usr/bin/env python3
"""
AI News Research Project
Main entry point for collecting and analyzing AI news
"""

from loguru import logger
import argparse
from datetime import datetime
import sys
from pathlib import Path
import yaml
from typing import Optional

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import configuration
from config import get_config, set_test_mode, set_bulk_mode

from src.scrapers.web_scraper import WebScraper
from src.scrapers.rss_scraper import RSSFeedScraper
from src.analyzers.nlp_analyzer import NLPAnalyzer
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.analyzers.genai_analyzer import GenAIAnalyzer
from src.analyzers.keyword_analyzer import KeywordAnalyzer
from src.models.database import Database


def setup_logging(config: dict):
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stdout,
        level=log_config.get('level', 'INFO'),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>"
    )
    
    # Add file logger
    log_file = str(log_config.get('file', 'logs/ai_news_research.log'))
    Path(log_file).parent.mkdir(exist_ok=True)
    
    logger.add(
        log_file,
        rotation=log_config.get('rotation', '1 week'),
        retention=log_config.get('retention', '1 month'),
        level=log_config.get('level', 'INFO')
    )


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from Python config module"""
    return get_config()


def collect_data(config: dict, filter_existing: bool = True) -> list:
    """Collect data from all configured sources"""
    all_articles = []
    data_config = config['data_collection']
    
    # Get existing URLs to avoid duplicates
    existing_urls = set()
    if filter_existing:
        try:
            db = Database(config['database'])
            existing_urls = db.get_existing_urls()
            logger.info(f"Found {len(existing_urls)} existing articles in database")
        except Exception as e:
            logger.warning(f"Could not load existing URLs: {e}")
    
    # Add keywords to each scraper config
    data_config['search_keywords'] = config['data_collection']['search_keywords']
    data_config['web_scraping']['search_keywords'] = config['data_collection']['search_keywords']
    data_config['reddit']['search_keywords'] = config['data_collection']['search_keywords']
    
    # Web scraping
    logger.info("Starting web scraping...")
    try:
        web_scraper = WebScraper(data_config['web_scraping'])
        web_articles = web_scraper.scrape_all_sites()
        # Filter out existing articles
        if filter_existing:
            new_web_articles = [a for a in web_articles if a.get('url') not in existing_urls]
            logger.info(f"Collected {len(web_articles)} articles from web scraping ({len(new_web_articles)} new)")
            all_articles.extend(new_web_articles)
        else:
            all_articles.extend(web_articles)
            logger.info(f"Collected {len(web_articles)} articles from web scraping")
    except Exception as e:
        logger.error(f"Web scraping failed: {str(e)}")
    
    # RSS feeds
    logger.info("Starting RSS feed collection...")
    try:
        rss_scraper = RSSFeedScraper(data_config)
        rss_articles = rss_scraper.scrape_all_feeds()
        # Filter out existing articles
        if filter_existing:
            new_rss_articles = [a for a in rss_articles if a.get('url') not in existing_urls]
            logger.info(f"Collected {len(rss_articles)} articles from RSS feeds ({len(new_rss_articles)} new)")
            all_articles.extend(new_rss_articles)
        else:
            all_articles.extend(rss_articles)
            logger.info(f"Collected {len(rss_articles)} articles from RSS feeds")
    except Exception as e:
        logger.error(f"RSS scraping failed: {str(e)}")
    
    # Reddit scraping disabled due to API changes
    logger.info("Reddit scraping is disabled (API incompatibility)")
    
    if filter_existing:
        logger.info(f"Total articles collected: {len(all_articles)} (new articles only)")
    else:
        logger.info(f"Total articles collected: {len(all_articles)}")
    return all_articles


def analyze_articles(articles: list, config: dict, test_genai: int = 0, test_mode: bool = False) -> list:
    """Perform NLP, sentiment, and GenAI analysis on articles"""
    from tqdm import tqdm
    
    # Limit articles in test mode
    if test_mode:
        original_count = len(articles)
        max_articles = config.get('test_mode', {}).get('max_articles_to_analyze', 5)
        articles = articles[:max_articles]
        logger.info(f"TEST MODE: Analyzing {len(articles)} articles (limited from {original_count})")
    else:
        logger.info(f"Analyzing {len(articles)} articles...")
    
    # Initialize analyzers and database
    nlp_analyzer = NLPAnalyzer(config['nlp'])
    sentiment_analyzer = SentimentAnalyzer(config['nlp'])
    genai_analyzer = GenAIAnalyzer(config['nlp'])
    db = Database(config['database'])
    
    # Check if GenAI API key is available
    genai_config = config['nlp'].get('genai', {})
    has_api_key = bool(genai_config.get('api_key', '').strip())
    
    # GenAI testing mode (only if API key is available)
    if test_genai > 0 and has_api_key:
        test_count = min(test_genai, len(articles))
        test_results = test_genai_analysis(articles[:test_count], genai_analyzer)
        if not confirm_full_analysis(test_results):
            logger.info("Full GenAI analysis cancelled by user")
            has_api_key = False  # Disable API usage but allow fallback
    
    # Analyze articles with progress bar and save each one
    analyzed_articles = []
    saved_count = 0
    
    with tqdm(total=len(articles), desc="Analyzing articles", unit="article") as pbar:
        for i, article in enumerate(articles):
            try:
                # NLP analysis
                article = nlp_analyzer.analyze_article(article)
                
                # Sentiment analysis
                article = sentiment_analyzer.analyze_article_sentiment(article)
                
                # GenAI analysis (always attempt - fallback if no API key)
                article = genai_analyzer.analyze_article(article)
                
                # Save article immediately after analysis
                try:
                    if db.save_article(article):
                        saved_count += 1
                    
                    # Mark as analyzed
                    if article.get('id'):
                        db.mark_articles_analyzed([article['id']])
                        
                except Exception as save_error:
                    logger.warning(f"Error saving article {i+1}: {save_error}")
                
                analyzed_articles.append(article)
                
                # Update progress bar with simple display
                pbar.update(1)
                if 'title' in article:
                    title = article['title'][:40] + "..." if len(article['title']) > 40 else article['title']
                    pbar.set_postfix_str(f"Current: {title}")
                
            except Exception as e:
                logger.error(f"Error analyzing article {i+1}: {e}")
                analyzed_articles.append(article)  # Add even failed articles
                pbar.update(1)
                pbar.set_postfix_str(f"Error on article {i+1}")
    
    # Log GenAI analysis statistics
    genai_stats = genai_analyzer.get_analysis_stats()
    if genai_stats['total'] > 0:
        if genai_stats['api_count'] > 0:
            logger.info(f"🤖 GenAI analysis: {genai_stats['api_count']} via API, {genai_stats['fallback_count']} via fallback")
        else:
            logger.info(f"📝 GenAI analysis: {genai_stats['fallback_count']} articles processed via fallback (no API key)")
    
    logger.info(f"Article analysis completed: {saved_count}/{len(articles)} articles saved to database")
    return analyzed_articles


def test_genai_analysis(test_articles: list, genai_analyzer) -> dict:
    """Test GenAI analysis on a small subset of articles"""
    from tqdm import tqdm
    import time
    
    logger.info(f"🧪 Testing GenAI analysis on {len(test_articles)} articles...")
    
    results = {
        'total': len(test_articles),
        'successful': 0,
        'failed': 0,
        'errors': [],
        'avg_time': 0,
        'avg_significance': 0,
        'start_time': time.time()
    }
    
    significance_scores = []
    
    with tqdm(total=len(test_articles), desc="Testing GenAI", unit="article") as pbar:
        for i, article in enumerate(test_articles):
            time.time()
            
            try:
                analyzed_article = genai_analyzer.analyze_article(article.copy())
                
                # Check if analysis was successful
                if 'genai_analysis' in analyzed_article:
                    results['successful'] += 1
                    
                    # Extract significance score
                    sig_score = analyzed_article.get('genai_analysis', {}).get('significance_score', 0)
                    if isinstance(sig_score, (int, float)) and 0 <= sig_score <= 1:
                        significance_scores.append(sig_score)
                else:
                    results['failed'] += 1
                    results['errors'].append(f"Article {i+1}: No GenAI analysis generated")
                    
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"Article {i+1}: {str(e)}")
            
            pbar.update(1)
            pbar.set_postfix_str(f"Success: {results['successful']}/{i+1}")
    
    # Calculate statistics
    total_time = time.time() - results['start_time']
    results['avg_time'] = total_time / len(test_articles)
    results['avg_significance'] = sum(significance_scores) / len(significance_scores) if significance_scores else 0
    
    # Print test results
    print("\n" + "="*50)
    print("🧪 GenAI TEST RESULTS")
    print("="*50)
    print(f"✅ Successful: {results['successful']}/{results['total']} ({results['successful']/results['total']*100:.1f}%)")
    print(f"❌ Failed: {results['failed']}/{results['total']}")
    print(f"⏱️  Average time: {results['avg_time']:.1f}s per article")
    print(f"📊 Average significance: {results['avg_significance']:.2f}")
    print(f"🕒 Total test time: {total_time:.1f}s")
    
    if results['errors']:
        print(f"\n❌ Errors encountered:")
        for error in results['errors'][:3]:  # Show first 3 errors
            print(f"  • {error}")
        if len(results['errors']) > 3:
            print(f"  • ... and {len(results['errors']) - 3} more errors")
    
    print("="*50)
    
    return results


def confirm_full_analysis(test_results: dict) -> bool:
    """Ask user confirmation to proceed with full analysis"""
    if test_results['successful'] == 0:
        print("❌ No articles were successfully analyzed. GenAI analysis will be disabled.")
        return False
    
    success_rate = test_results['successful'] / test_results['total'] * 100
    
    if success_rate >= 80:
        print(f"✅ Test passed with {success_rate:.1f}% success rate. Proceeding with full analysis...")
        return True
    elif success_rate >= 50:
        response = input(f"\n⚠️  Test success rate is {success_rate:.1f}%. Continue with full analysis? [y/N]: ")
        return response.lower().startswith('y')
    else:
        print(f"❌ Test success rate is only {success_rate:.1f}%. GenAI analysis will be disabled.")
        return False


def save_to_database(articles: list, config: dict) -> int:
    """Save articles to database and mark them as analyzed"""
    logger.info("Saving articles to database...")
    
    db = Database(config['database'])
    saved_count = db.save_articles(articles)
    
    # Mark all saved articles as analyzed (since they went through analysis)
    article_ids = [article.get('id') for article in articles if article.get('id')]
    if article_ids:
        db.mark_articles_analyzed(article_ids)
    
    logger.info(f"Saved {saved_count} articles to database and marked {len(article_ids)} as analyzed")
    return saved_count


def generate_report(config: dict):
    """Generate analysis report"""
    logger.info("Generating analysis report...")
    
    db = Database(config['database'])
    nlp_analyzer = NLPAnalyzer(config['nlp'])
    sentiment_analyzer = SentimentAnalyzer(config['nlp'])
    
    # Get all articles from database
    articles = db.get_articles(limit=1000)
    
    # Perform corpus analysis
    nlp_results = nlp_analyzer.analyze_corpus(articles)
    sentiment_results = sentiment_analyzer.analyze_corpus_sentiment(articles)
    
    # Get database statistics
    db_stats = db.get_statistics()
    
    # Create report
    report = {
        'generated_at': datetime.now().isoformat(),
        'database_stats': db_stats,
        'nlp_analysis': nlp_results,
        'sentiment_analysis': sentiment_results
    }
    
    # Save report
    report_dir = Path(config['output']['reports_dir'])
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / f"ai_news_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    
    with open(report_file, 'w') as f:
        yaml.dump(report, f, default_flow_style=False)
    
    logger.info(f"Report saved to {report_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("AI NEWS ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total articles: {db_stats['total_articles']}")
    print(f"Sources: {db_stats['sources']}")
    print(f"Date range: {db_stats['date_range']['earliest']} to {db_stats['date_range']['latest']}")
    print("\nSentiment Distribution:")
    for sentiment, percentage in sentiment_results['overall_distribution'].items():
        print(f"  {sentiment.capitalize()}: {percentage:.1f}%")
    print("\nTop Themes:")
    if 'theme_distribution' in nlp_results:
        for theme, count in sorted(nlp_results['theme_distribution'].items(), 
                                  key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {theme}: {count} articles")
    print("="*50)


def analyze_keywords(config: dict, days_back: Optional[int] = None):
    """Analyze and export keywords from articles in database"""
    logger.info("Starting keyword analysis...")
    
    db = Database(config['database'])
    keyword_analyzer = KeywordAnalyzer(config)
    
    # Analyze keywords
    results = keyword_analyzer.analyze_keywords_from_database(
        database=db,
        min_frequency=3,
        top_n=200,
        days_back=days_back
    )
    
    if results['status'] == 'success':
        # Save keywords
        output_path = keyword_analyzer.save_keywords(results)
        
        # Print summary
        print("\n" + "="*50)
        print("KEYWORD ANALYSIS RESULTS")
        print("="*50)
        print(f"Total articles analyzed: {results['statistics']['total_articles']}")
        print(f"Total unique terms: {results['statistics']['total_unique_terms']}")
        if results['statistics']['date_range']:
            print(f"Date range: {results['statistics']['date_range']['earliest']} to {results['statistics']['date_range']['latest']}")
        print(f"\nTop 20 Keywords:")
        for i, kw in enumerate(results['keywords'][:20], 1):
            sources = ', '.join(kw['sources'])
            ai_marker = " 🤖" if kw['is_ai_term'] else ""
            print(f"  {i:2d}. {kw['keyword']:<30} (score: {kw['score']:.3f}, freq: {kw['frequency']}, sources: {sources}){ai_marker}")
        
        print(f"\nKeywords saved to:")
        print(f"  - {output_path} (YAML format for editing)")
        print(f"  - {Path(output_path).with_suffix('.json')} (JSON format)")
        print(f"  - {Path(output_path).with_suffix('.txt')} (Simple text list)")
        print("\nEdit the files as needed to customize your keyword list.")
        print("="*50)
    else:
        logger.error(f"Keyword analysis failed: {results.get('message', 'Unknown error')}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='AI News Research Project')
    parser.add_argument(
        '--config', 
        default=None,
        help='Configuration parameter (unused, kept for compatibility)'
    )
    parser.add_argument(
        '--collect',
        action='store_true',
        help='Collect new data from sources'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze collected data'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate analysis report'
    )
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Launch interactive dashboard'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode with reduced limits'
    )
    parser.add_argument(
        '--bulk',
        action='store_true',
        help='Run in bulk mode for collecting large amounts of data'
    )
    parser.add_argument(
        '--multi-run',
        type=int,
        default=1,
        help='Run collection multiple times (useful for accumulating data)'
    )
    parser.add_argument(
        '--test-genai',
        type=int,
        default=0,
        help='Test GenAI analysis on N articles before full analysis'
    )
    parser.add_argument(
        '--reanalyze',
        action='store_true',
        help='Reanalyze all articles including previously analyzed ones'
    )
    parser.add_argument(
        '--keywords',
        action='store_true',
        help='Analyze articles to extract and export keywords'
    )
    parser.add_argument(
        '--keywords-days',
        type=int,
        default=None,
        help='Analyze keywords from last N days only (default: all articles)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.test:
        logger.info("Running in TEST MODE with reduced limits")
        config = set_test_mode()
    elif args.bulk:
        logger.info("Running in BULK MODE for maximum data collection")
        config = set_bulk_mode()
    else:
        config = load_config()
    
    setup_logging(config)
        
    logger.info("AI News Research Project started")
    
    try:
        if args.collect:
            # Multi-run collection
            total_new_articles = 0
            for run in range(args.multi_run):
                if args.multi_run > 1:
                    logger.info(f"Collection run {run + 1}/{args.multi_run}")
                
                # Collect data
                articles = collect_data(config, filter_existing=not args.reanalyze)
                
                # Analyze articles (articles are saved individually during analysis)
                articles = analyze_articles(articles, config, args.test_genai, args.test)
                
                # Count new articles (articles already saved during analysis)
                new_count = len([a for a in articles if a.get('id')])
                total_new_articles += new_count
                
                if run < args.multi_run - 1:
                    logger.info(f"Waiting 5 seconds before next run...")
                    import time
                    time.sleep(5)
            
            if args.multi_run > 1:
                logger.info(f"Multi-run completed. Total new articles: {total_new_articles}")
            
        elif args.analyze:
            # Analyze existing data in database
            db = Database(config['database'])
            
            if args.reanalyze:
                # Reanalyze all articles
                articles = db.get_articles(limit=1000)
                logger.info("Reanalyzing all articles in database")
            else:
                # Only analyze unanalyzed articles
                articles = db.get_unanalyzed_articles(limit=1000)
                logger.info("Analyzing only unanalyzed articles")
            
            if not articles:
                if args.reanalyze:
                    logger.warning("No articles found in database. Run with --collect first.")
                else:
                    logger.info("No unanalyzed articles found. All articles have been analyzed.")
                return
            
            # Analyze articles (articles are saved individually during analysis)
            articles = analyze_articles(articles, config, args.test_genai, args.test)
            
        elif args.report:
            # Generate report
            generate_report(config)
            
        elif args.dashboard:
            # Launch dashboard
            logger.info("Launching dashboard...")
            import subprocess
            import sys
            
            try:
                # Launch Streamlit dashboard
                subprocess.run([
                    sys.executable, '-m', 'streamlit', 'run', 'dashboard.py',
                    '--server.headless', 'true',
                    '--server.address', '0.0.0.0',
                    '--server.port', '8501'
                ])
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to launch dashboard: {e}")
                logger.info("You can manually run: streamlit run dashboard.py")
        
        elif args.keywords:
            # Analyze and export keywords
            analyze_keywords(config, days_back=args.keywords_days)
            
        else:
            # Default: collect, analyze, and generate report
            logger.info("Running full pipeline: collect -> analyze -> report")
            
            # Collect
            articles = collect_data(config, filter_existing=not args.reanalyze)
            
            # Analyze (articles are saved individually during analysis)
            articles = analyze_articles(articles, config, args.test_genai, args.test)
            
            # Report
            generate_report(config)
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise
    finally:
        logger.info("AI News Research Project completed")


if __name__ == '__main__':
    main()