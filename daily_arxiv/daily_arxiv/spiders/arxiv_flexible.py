# -*- coding: utf-8 -*-
import os
import time
import scrapy
import requests
import argparse
from datetime import datetime, timedelta
from urllib.parse import urlencode
from ..utils import DateValidator, validate_keywords, generate_search_query

class ArxivFlexibleSpider(scrapy.Spider):
    name = "arxiv_flexible"

    def __init__(self, start_date=None, end_date=None, *args, **kwargs):
        super(ArxivFlexibleSpider, self).__init__(*args, **kwargs)
        self.start_date = start_date
        self.end_date = end_date

    def wait_until_data_ready(self, target_date):
        """等待指定日期的数据更新完成"""
        check_interval = int(os.environ.get("CHECK_INTERVAL", 1800))  # 30 分钟
        max_retry = int(os.environ.get("MAX_RETRY", 12))             # 默认 6 小时
        base_url = "https://export.arxiv.org/api/query?"

        for attempt in range(1, max_retry + 1):
            if self.is_date_data_ready(base_url, target_date):
                self.logger.info(f"✅ {target_date} 数据已更新，开始爬取。")
                return True
            else:
                self.logger.warning(f"⚠️ 第 {attempt} 次检测：{target_date} 数据尚未更新，{check_interval//60} 分钟后重试...")
                time.sleep(check_interval)

        self.logger.error(f"❌ 超过最大重试次数，{target_date} 数据仍未更新，终止爬取。")
        return False

    def is_date_data_ready(self, base_url: str, target_date) -> bool:
        """检查指定日期是否至少有一条论文数据"""
        params = {
            "search_query": "all",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": 50,  # 检查最近 50 条
        }
        try:
            url = base_url + urlencode(params)
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            # 判断指定日期是否出现在最新结果中
            if str(target_date) in resp.text:
                self.logger.info(f"检测到 {target_date} 出现在最新结果中")
                return True
            return False
        except Exception as e:
            self.logger.warning(f"检测 {target_date} 数据失败：{e}")
            return False

    def parse_date_input(self, date_str):
        """解析日期输入，支持多种格式"""
        parsed_date = DateValidator.parse_date_input(date_str)
        if parsed_date:
            return parsed_date.date()
        else:
            self.logger.error(f"无效的日期格式: {date_str}")
            return None

    def start_requests(self):
        # === 获取并验证关键词 ===
        keywords_str = os.environ.get('KEYWORDS', '')
        is_valid, error_msg, keywords = validate_keywords(keywords_str)
        if not is_valid:
            self.logger.error(f"⚠️ 关键词验证失败: {error_msg}")
            return
        self.logger.info(f"关键词: {keywords}")
        self.search_query = generate_search_query(keywords)

        # === 解析并验证日期范围 ===
        start_date_str = self.start_date or os.environ.get('START_DATE', 'today')
        end_date_str = self.end_date or os.environ.get('END_DATE', 'today')
        
        is_valid, error_msg, parsed_start, parsed_end = DateValidator.validate_date_range(start_date_str, end_date_str)
        if not is_valid:
            self.logger.error(f"❌ 日期范围验证失败: {error_msg}")
            return
        
        self.start_date = parsed_start.date()
        self.end_date = parsed_end.date()
        
        # 确保开始日期不晚于结束日期
        if self.start_date > self.end_date:
            self.start_date, self.end_date = self.end_date, self.start_date
            
        date_range_desc = DateValidator.get_date_range_description(start_date_str, end_date_str)
        self.logger.info(f"爬取日期范围: {date_range_desc}")

        # 检查是否需要等待数据更新（如果爬取的是今天或未来的日期）
        today = datetime.utcnow().date()
        if self.start_date >= today:
            if not self.wait_until_data_ready(self.start_date):
                return

        # === 请求参数 ===
        self.base_url = "https://export.arxiv.org/api/query?"
        self.per_page = int(os.environ.get('PER_PAGE', 200))
        self.max_pages = int(os.environ.get('MAX_PAGES', 10))
        self.date_field = os.environ.get('DATE_FIELD', 'published').strip().lower()
        if self.date_field not in ('published', 'updated'):
            self.date_field = 'published'

        params = {
            'search_query': self.search_query,
            'start': 0,
            'max_results': self.per_page,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        url = self.base_url + urlencode(params)
        yield scrapy.Request(url, callback=self.parse_api_response, meta={'start': 0, 'page': 1})

    def parse_api_response(self, response):
        start = response.meta.get('start', 0)
        page = response.meta.get('page', 1)

        entries = response.xpath('//*[local-name()="entry"]')
        if not entries:
            self.logger.warning(f"第 {page} 页未返回 entry")
            return

        found_in_page = 0
        stop_paging = False
        for entry in entries:
            date_text = entry.xpath(f'*[local-name()="{self.date_field}"]/text()').get()
            if not date_text:
                continue
            try:
                pub_dt = datetime.fromisoformat(date_text.replace('Z', '+00:00'))
            except Exception:
                continue
            pub_date = pub_dt.date()
            
            # 检查日期是否在目标范围内
            if pub_date > self.end_date:
                continue
            elif self.start_date <= pub_date <= self.end_date:
                found_in_page += 1
                arxiv_id_url = entry.xpath('*[local-name()="id"]/text()').get() or ''
                arxiv_id = arxiv_id_url.split('/')[-1] if arxiv_id_url else None
                
                # 获取分类信息
                categories = entry.xpath('*[local-name()="category"]/@term').getall()
                
                yield {
                    "id": arxiv_id,
                    "title": entry.xpath('*[local-name()="title"]/text()').get(),
                    "summary": entry.xpath('*[local-name()="summary"]/text()').get(),
                    "authors": entry.xpath('*[local-name()="author"]/*[local-name()="name"]/text()').getall(),
                    "categories": categories,
                    self.date_field: date_text,
                    "pdf": f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None,
                    "abs": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None
                }
            elif pub_date < self.start_date:
                stop_paging = True
                break

        self.logger.info(f"第 {page} 页找到 {found_in_page} 条目标日期论文")

        # 翻页
        if not stop_paging and len(entries) == self.per_page:
            next_start = start + self.per_page
            next_page = page + 1
            if next_page > self.max_pages:
                self.logger.warning(f"已到达 max_pages，停止翻页")
                return
            next_url = self.base_url + urlencode({
                'search_query': self.search_query,
                'start': next_start,
                'max_results': self.per_page,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            })
            yield scrapy.Request(next_url, callback=self.parse_api_response, meta={'start': next_start, 'page': next_page})
        else:
            self.logger.info("翻页结束，爬取完成")
