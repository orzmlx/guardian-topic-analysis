"""
The Guardian 新闻爬虫
====================
使用 Guardian Open Platform API 获取 2008-2026 年新闻数据
API 文档: https://open-platform.theguardian.com/documentation/

Guardian API 免费版每天 5000 次请求，足够获取大量数据
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
    """The Guardian 新闻爬虫"""
    
    # Guardian 开放 API - 免费 test key（每天 500 次）
    # 如需更多请求，可在 https://bonobo.capi.gutools.co.uk/register/developer 注册获取免费 key
    DEFAULT_API_KEY = "test"  # 测试用 key
    
    def __init__(self, api_key=None):
        self.api_key = api_key or self.DEFAULT_API_KEY
        self.base_url = "https://content.guardianapis.com/search"
        self.session = requests.Session()
        self.articles = []
    
    def fetch_articles(self, from_date, to_date, section=None, page_size=200, max_pages=50):
        """
        从 Guardian API 获取文章
        
        参数:
            from_date: 开始日期 (YYYY-MM-DD)
            to_date: 结束日期 (YYYY-MM-DD)
            section: 新闻分类 (world, uk-news, business, technology, etc.)
            page_size: 每页结果数 (最大 200)
            max_pages: 最大页数
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
                        logger.info(f"  已获取 {len(all_articles)} 条 (页 {page}/{min(total_pages, max_pages)})")
                    
                elif resp.status_code == 429:
                    logger.warning("API 限流，等待 60 秒...")
                    time.sleep(60)
                    continue
                else:
                    logger.warning(f"API 错误: {resp.status_code}")
                    break
                    
            except Exception as e:
                logger.error(f"请求失败: {e}")
                break
            
            page += 1
            time.sleep(random.uniform(0.2, 0.5))  # 避免请求过快
        
        return all_articles
    
    def _format_date(self, iso_date):
        """将 ISO 日期转换为与 BBC 相同的格式"""
        try:
            dt = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
            return dt.strftime('%a, %d %b %Y %H:%M:%S GMT')
        except:
            return iso_date
    
    def fetch_by_year(self, year, sections=None):
        """获取指定年份的新闻"""
        if sections is None:
            sections = [
                'world', 'uk-news', 'business', 'technology',
                'politics', 'environment', 'science', 'media'
            ]
        
        all_articles = []
        
        for section in sections:
            logger.info(f"获取 {year} 年 {section} 新闻...")
            
            articles = self.fetch_articles(
                from_date=f"{year}-01-01",
                to_date=f"{year}-12-31",
                section=section,
                page_size=200,
                max_pages=25  # 每个分类最多 5000 条
            )
            
            all_articles.extend(articles)
            logger.info(f"  {section}: {len(articles)} 条")
            
            time.sleep(1)
        
        return all_articles
    
    def fetch_all_years(self, start_year=2008, end_year=2026):
        """获取多年数据"""
        all_articles = []
        
        # 重要新闻分类
        main_sections = ['world', 'uk-news', 'business', 'politics']
        other_sections = ['technology', 'environment', 'science']
        
        for year in range(start_year, end_year + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"获取 {year} 年新闻...")
            logger.info(f"{'='*50}")
            
            # 主要分类获取更多
            for section in main_sections:
                articles = self.fetch_articles(
                    from_date=f"{year}-01-01",
                    to_date=f"{year}-12-31",
                    section=section,
                    page_size=200,
                    max_pages=15
                )
                all_articles.extend(articles)
                logger.info(f"  {section}: {len(articles)} 条")
                time.sleep(0.5)
            
            # 其他分类
            for section in other_sections:
                articles = self.fetch_articles(
                    from_date=f"{year}-01-01",
                    to_date=f"{year}-12-31",
                    section=section,
                    page_size=200,
                    max_pages=5
                )
                all_articles.extend(articles)
                logger.info(f"  {section}: {len(articles)} 条")
                time.sleep(0.5)
            
            year_count = sum(1 for a in all_articles if str(year) in a.get('pubDate', ''))
            logger.info(f"{year} 年共获取约 {year_count} 条")
        
        return all_articles


class GuardianRSSScraper:
    """通过 RSS 获取 Guardian 最新新闻"""
    
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
        """获取所有 RSS 源的新闻"""
        all_articles = []
        
        for section, rss_url in self.RSS_FEEDS.items():
            logger.info(f"获取 RSS: {section}")
            
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
                logger.warning(f"RSS 获取失败 {section}: {e}")
            
            time.sleep(0.5)
        
        logger.info(f"RSS 共获取 {len(all_articles)} 条")
        return all_articles


def main():
    output_dir = '/Users/liuxi/Desktop/text mining'
    
    print("=" * 60)
    print("The Guardian 新闻爬虫")
    print("=" * 60)
    
    # 1. 使用 API 获取历史数据
    scraper = GuardianScraper()
    
    all_articles = []
    
    # 关键年份：获取更多数据
    key_years = [2008, 2009, 2019, 2020, 2021]
    other_years = [y for y in range(2010, 2019)] + [2022, 2023, 2024, 2025, 2026]
    
    # 获取关键年份（每年约 3000-5000 条）
    logger.info("\n获取关键年份数据...")
    for year in key_years:
        logger.info(f"\n{'='*40}")
        logger.info(f"获取 {year} 年（关键年份）")
        
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
            logger.info(f"  {year} {section}: {len(articles)} 条")
            time.sleep(0.3)
    
    # 获取其他年份（每年约 1000-2000 条）
    logger.info("\n获取其他年份数据...")
    for year in other_years:
        logger.info(f"\n获取 {year} 年")
        
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
        logger.info(f"  {year} 年: ~{count} 条")
    
    # 2. 使用 RSS 获取最新数据
    logger.info("\n获取 RSS 最新新闻...")
    try:
        rss_scraper = GuardianRSSScraper()
        rss_articles = rss_scraper.fetch_rss()
        all_articles.extend(rss_articles)
    except Exception as e:
        logger.warning(f"RSS 获取失败: {e}")
    
    # 3. 去重
    seen_titles = set()
    unique_articles = []
    for art in all_articles:
        title = art.get('title', '')
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_articles.append(art)
    
    logger.info(f"\n去重后: {len(unique_articles)} 条")
    
    # 4. 保存为 CSV（与 bbc_news.csv 结构相同）
    df = pd.DataFrame(unique_articles)
    
    # 确保列顺序与 bbc_news.csv 相同
    cols = ['title', 'pubDate', 'guid', 'link', 'description']
    df = df[[c for c in cols if c in df.columns]]
    
    # 保存
    output_path = os.path.join(output_dir, 'guardian_news.csv')
    df.to_csv(output_path, index=False)
    
    # 5. 统计信息
    print("\n" + "=" * 60)
    print("✅ Guardian 新闻爬取完成!")
    print("=" * 60)
    print(f"总计: {len(df)} 条新闻")
    print(f"保存到: {output_path}")
    
    # 按年份统计
    def extract_year(date_str):
        match = re.search(r'(20\d{2})', str(date_str))
        return int(match.group(1)) if match else None
    
    df['year'] = df['pubDate'].apply(extract_year)
    
    print("\n按年份分布:")
    year_counts = df['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        if year and 2008 <= year <= 2026:
            bar = '█' * min(int(count/100), 40)
            print(f"  {int(year)}: {count:>5} {bar}")
    
    return df


if __name__ == "__main__":
    main()
