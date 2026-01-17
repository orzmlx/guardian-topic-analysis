"""
The Guardian News Scraper
=========================
Fetch news data from 2008-2026 using The Guardian Open Platform API.
API Documentation: https://open-platform.theguardian.com/documentation/

The free version of Guardian API allows 5000 requests per day, which is sufficient for fetching a large amount of data.
"""

import requests
import pandas as pd
import time
import random
import re
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class GuardianScraper:
    """The Guardian News Scraper"""
    
    # Guardian Open API - Free test key (500 requests per day)
    # For more requests, register for a free key at https://bonobo.capi.gutools.co.uk/register/developer
    DEFAULT_API_KEY = "test"  # Test key
    
    def __init__(self, api_key=None):
        self.api_key = api_key or self.DEFAULT_API_KEY
        self.base_url = "https://content.guardianapis.com/search"
        self.session = requests.Session()
        self.articles = []
    
    def fetch_articles(self, from_date, to_date, section=None, page_size=200, max_pages=50):
        """
        Fetch articles from Guardian API
        
        Args:
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            section: News section (world, uk-news, business, technology, etc.)
            page_size: Number of results per page (max 200)
            max_pages: Maximum number of pages to fetch
        """
        all_articles = []
        
        params = {
            'api-key': self.api_key,
            'from-date': from_date,
            'to-date': to_date,
            'page-size': page_size,
            'show-fields': 'headline,trailText,shortUrl,publication',
            'order-by': 'newest',
        }
        
        if section:
            params['section'] = section
        
        page = 1
        total_pages = 1
        
        while page <= min(total_pages, max_pages):
            params['page'] = page
            
            try:
                resp = self.session.get(self.base_url, params=params, timeout=30)
                
                if resp.status_code == 200:
                    data = resp.json()
                    response = data.get('response', {})
                    
                    total_pages = response.get('pages', 1)
                    results = response.get('results', [])
                    
                    for item in results:
                        fields = item.get('fields', {})
                        
                        article = {
                            'title': fields.get('headline', item.get('webTitle', '')),
                            'pubDate': self._format_date(item.get('webPublicationDate', '')),
                            'guid': item.get('id', ''),
                            'link': item.get('webUrl', ''),
                            'description': fields.get('trailText', '')[:500] if fields.get('trailText') else '',
                        }
                        all_articles.append(article)
                    
                    if page % 10 == 0:
                        logger.info(f"  Fetched {len(all_articles)} articles (Page {page}/{min(total_pages, max_pages)})")
                    
                elif resp.status_code == 429:
                    logger.warning("API rate limit exceeded, waiting for 60 seconds...")
                    time.sleep(60)
                    continue
                else:
                    logger.warning(f"API Error: {resp.status_code}")
                    break
                    
            except Exception as e:
                logger.error(f"Request failed: {e}")
                break
            
            page += 1
            time.sleep(random.uniform(0.2, 0.5))  # Avoid requesting too fast
        
        return all_articles
    
    def _format_date(self, iso_date):
        """Convert ISO date to the same format as BBC (RFC 2822)"""
        try:
            dt = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
            return dt.strftime('%a, %d %b %Y %H:%M:%S GMT')
        except:
            return iso_date
    
    def fetch_by_year(self, year, sections=None):
        """Fetch news for a specific year"""
        if sections is None:
            sections = [
                'world', 'uk-news', 'business', 'technology',
                'politics', 'environment', 'science', 'media'
            ]
        
        all_articles = []
        
        for section in sections:
            logger.info(f"Fetching {section} news for {year}...")
            
            articles = self.fetch_articles(
                from_date=f"{year}-01-01",
                to_date=f"{year}-12-31",
                section=section,
                page_size=200,
                max_pages=25  # Max 5000 articles per section
            )
            
            all_articles.extend(articles)
            logger.info(f"  {section}: {len(articles)} articles")
            
            time.sleep(1)
        
        return all_articles
    
    def fetch_all_years(self, start_year=2008, end_year=2026):
        """Fetch data for multiple years"""
        all_articles = []
        
        # Main news sections
        main_sections = ['world', 'uk-news', 'business', 'politics']
        other_sections = ['technology', 'environment', 'science']
        
        for year in range(start_year, end_year + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Fetching news for {year}...")
            logger.info(f"{'='*50}")
            
            # Fetch more for main sections
            for section in main_sections:
                articles = self.fetch_articles(
                    from_date=f"{year}-01-01",
                    to_date=f"{year}-12-31",
                    section=section,
                    page_size=200,
                    max_pages=15
                )
                all_articles.extend(articles)
                logger.info(f"  {section}: {len(articles)} articles")
                time.sleep(0.5)
            
            # Other sections
            for section in other_sections:
                articles = self.fetch_articles(
                    from_date=f"{year}-01-01",
                    to_date=f"{year}-12-31",
                    section=section,
                    page_size=200,
                    max_pages=5
                )
                all_articles.extend(articles)
                logger.info(f"  {section}: {len(articles)} articles")
                time.sleep(0.5)
            
            year_count = sum(1 for a in all_articles if str(year) in a.get('pubDate', ''))
            logger.info(f"Fetched approximately {year_count} articles for {year}")
        
        return all_articles


class GuardianRSSScraper:
    """Fetch latest Guardian news via RSS"""
    
    RSS_FEEDS = {
        'world': 'https://www.theguardian.com/world/rss',
        'uk': 'https://www.theguardian.com/uk-news/rss',
        'business': 'https://www.theguardian.com/uk/business/rss',
        'technology': 'https://www.theguardian.com/uk/technology/rss',
        'politics': 'https://www.theguardian.com/politics/rss',
        'environment': 'https://www.theguardian.com/environment/rss',
        'science': 'https://www.theguardian.com/science/rss',
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
        })
    
    def fetch_rss(self):
        """Fetch news from all RSS feeds"""
        all_articles = []
        
        for section, rss_url in self.RSS_FEEDS.items():
            logger.info(f"Fetching RSS: {section}")
            
            try:
                resp = self.session.get(rss_url, timeout=30)
                if resp.status_code == 200:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(resp.content, 'xml')
                    
                    for item in soup.find_all('item'):
                        article = {
                            'title': item.find('title').text if item.find('title') else '',
                            'pubDate': item.find('pubDate').text if item.find('pubDate') else '',
                            'guid': item.find('guid').text if item.find('guid') else '',
                            'link': item.find('link').text if item.find('link') else '',
                            'description': item.find('description').text[:500] if item.find('description') else '',
                        }
                        all_articles.append(article)
                        
            except Exception as e:
                logger.warning(f"Failed to fetch RSS {section}: {e}")
            
            time.sleep(0.5)
        
        logger.info(f"Fetched {len(all_articles)} articles via RSS")
        return all_articles


def main():
    output_dir = '/Users/liuxi/Desktop/text mining'
    
    print("=" * 60)
    print("The Guardian News Scraper")
    print("=" * 60)
    
    # 1. Fetch historical data using API
    scraper = GuardianScraper()
    
    all_articles = []
    
    # Key years: fetch more data
    key_years = [2008, 2009, 2019, 2020, 2021]
    other_years = [y for y in range(2010, 2019)] + [2022, 2023, 2024, 2025, 2026]
    
    # Fetch key years (approx. 3000-5000 articles per year)
    logger.info("\nFetching data for key years...")
    for year in key_years:
        logger.info(f"\n{'='*40}")
        logger.info(f"Fetching {year} (Key Year)")
        
        sections = ['world', 'uk-news', 'business', 'politics', 'technology', 'environment']
        
        for section in sections:
            articles = scraper.fetch_articles(
                from_date=f"{year}-01-01",
                to_date=f"{year}-12-31",
                section=section,
                page_size=200,
                max_pages=20
            )
            all_articles.extend(articles)
            logger.info(f"  {year} {section}: {len(articles)} articles")
            time.sleep(0.3)
    
    # Fetch other years (approx. 1000-2000 articles per year)
    logger.info("\nFetching data for other years...")
    for year in other_years:
        logger.info(f"\nFetching {year}")
        
        sections = ['world', 'uk-news', 'business', 'politics']
        
        for section in sections:
            articles = scraper.fetch_articles(
                from_date=f"{year}-01-01",
                to_date=f"{year}-12-31",
                section=section,
                page_size=200,
                max_pages=10
            )
            all_articles.extend(articles)
            time.sleep(0.3)
        
        count = len([a for a in all_articles if str(year) in a.get('pubDate', '')])
        logger.info(f"  {year}: ~{count} articles")
    
    # 2. Fetch latest data using RSS
    logger.info("\nFetching latest news via RSS...")
    try:
        rss_scraper = GuardianRSSScraper()
        rss_articles = rss_scraper.fetch_rss()
        all_articles.extend(rss_articles)
    except Exception as e:
        logger.warning(f"RSS fetch failed: {e}")
    
    # 3. Deduplication
    seen_titles = set()
    unique_articles = []
    for art in all_articles:
        title = art.get('title', '')
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_articles.append(art)
    
    logger.info(f"\nAfter deduplication: {len(unique_articles)} articles")
    
    # 4. Save to CSV (same structure as bbc_news.csv)
    df = pd.DataFrame(unique_articles)
    
    # Ensure column order matches bbc_news.csv
    cols = ['title', 'pubDate', 'guid', 'link', 'description']
    df = df[[c for c in cols if c in df.columns]]
    
    # Save
    output_path = os.path.join(output_dir, 'guardian_news.csv')
    df.to_csv(output_path, index=False)
    
    # 5. Statistics
    print("\n" + "=" * 60)
    print("✅ Guardian News Fetching Complete!")
    print("=" * 60)
    print(f"Total: {len(df)} articles")
    print(f"Saved to: {output_path}")
    
    # Statistics by year
    def extract_year(date_str):
        match = re.search(r'(20\d{2})', str(date_str))
        return int(match.group(1)) if match else None
    
    df['year'] = df['pubDate'].apply(extract_year)
    
    print("\nDistribution by Year:")
    year_counts = df['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        if year and 2008 <= year <= 2026:
            bar = '█' * min(int(count/100), 40)
            print(f"  {int(year)}: {count:>5} {bar}")
    
    return df


if __name__ == "__main__":
    main()
