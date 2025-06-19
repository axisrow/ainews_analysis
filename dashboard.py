#!/usr/bin/env python3
"""
AI News Research Dashboard
Streamlit dashboard for visualizing AI news analysis results
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from config import get_config
from visualization.dashboard import run_dashboard


def main():
    """Main entry point for dashboard"""
    # Load configuration
    config = get_config()
    
    # Run dashboard
    run_dashboard(config)


if __name__ == "__main__":
    main()