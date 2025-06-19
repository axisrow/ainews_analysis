#!/usr/bin/env python3
"""
Test script to verify dashboard functionality
"""

import yaml
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from models.database import Database


def test_dashboard_data():
    """Test if dashboard has data to display"""
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test database connection
    db = Database(config['database'])
    
    # Get articles
    articles = db.get_articles(limit=10)
    print(f"✅ Database connection: OK")
    print(f"✅ Articles in database: {len(articles)}")
    
    if articles:
        print(f"✅ Sample article: {articles[0]['title'][:50]}...")
        print(f"✅ Sources available: {len(db.get_sources())}")
        print(f"✅ Article count: {db.get_article_count()}")
    else:
        print("❌ No articles found. Run 'python main.py --collect' first.")
        return False
    
    return True


def test_dashboard_import():
    """Test if dashboard modules can be imported"""
    try:
        from visualization.dashboard import run_dashboard
        print("✅ Dashboard import: OK")
        return True
    except ImportError as e:
        print(f"❌ Dashboard import failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing AI News Dashboard...")
    print("=" * 40)
    
    success = True
    success &= test_dashboard_import()
    success &= test_dashboard_data()
    
    print("=" * 40)
    if success:
        print("🎉 All tests passed! Dashboard should work correctly.")
        print("\n📊 To launch dashboard:")
        print("   streamlit run dashboard.py")
        print("   or")
        print("   python main.py --dashboard")
    else:
        print("❌ Some tests failed. Check the errors above.")