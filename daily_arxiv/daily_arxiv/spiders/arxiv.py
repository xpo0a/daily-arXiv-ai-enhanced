# -*- coding: utf-8 -*-
import os
import time
import scrapy
import requests
from datetime import datetime, timedelta
from urllib.parse import urlencode

class ArxivSpider(scrapy.Spider):
    name = "arxiv_weekly"

    def wait_until_data_ready(self):
        """等待上周数据更新完成"""
        check_interval = int(os.environ.get("CHECK_INTERVAL", 1800))  # 30 分钟
        max_retry = int(os.environ.get("MAX_RETRY", 12))             # 默认 6 小时
        base_url = "https://export.arxiv.org/api/query?"

        for attempt in range(1, max_retry + 1):
            if self.is_last_week_data_ready(base_url):
                self.logger.info("✅ 上周数据已更新，开始爬取。")
                return True
            else:
                self.logger.warning(f"⚠️ 第 {attempt} 次检测：上周数据尚未更新，{check_interval//60} 分钟后重试...")
                time.sleep(check_interval)

        self.logger.error("❌ 超过最大重试次数，上周数据仍未更新，终止爬取。")
        return False

    def is_last_week_data_ready(self, base_url: str) -> bool:
        """检查上周是否至少有一条论文数据"""
        today = datetime.utcnow().date()
        last_monday = today - timedelta(days=today.weekday() + 7)
        last_sunday = last_monday + timedelta(days=6)

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
            # 判断上周任意一天是否出现
            for i in range(7):
                day = last_monday + timedelta(days=i)
                if str(day) in resp.text:
                    self.logger.info(f"检测到上周 {day} 出现在最新结果中")
                    return True
            return False
        except Exception as e:
            self.logger.warning(f"检测上周数据失败：{e}")
            return False

    def start_requests(self):
        if not self.wait_until_data_ready():
            return  # 数据未准备好则退出

        # === 获取关键词 ===
        keywords_str = os.environ.get('KEYWORDS', '')
        keywords = [k.strip().strip('"').strip("'") for k in keywords_str.split(',') if k.strip()]
        if not keywords:
            self.logger.error("⚠️ KEYWORDS 为空或格式不正确")
            return
        self.logger.info(f"关键词: {keywords}")
        self.search_query = ' OR '.join([f'(ti:"{k}" OR abs:"{k}")' for k in keywords])

        # === 上周时间范围 ===
        today = datetime.utcnow().date()
        self.start_date = today - timedelta(days=today.weekday() + 7)  # 上周一
        self.end_date = self.start_date + timedelta(days=6)             # 上周日
        self.logger.info(f"爬取上周论文: {self.start_date} ~ {self.end_date}")

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
            if pub_date > self.end_date:
                continue
            elif self.start_date <= pub_date <= self.end_date:
                found_in_page += 1
                arxiv_id_url = entry.xpath('*[local-name()="id"]/text()').get() or ''
                arxiv_id = arxiv_id_url.split('/')[-1] if arxiv_id_url else None
                yield {
                    "id": arxiv_id,
                    "title": entry.xpath('*[local-name()="title"]/text()').get(),
                    "summary": entry.xpath('*[local-name()="summary"]/text()').get(),
                    "authors": entry.xpath('*[local-name()="author"]/*[local-name()="name"]/text()').getall(),
                    self.date_field: date_text,
                    "pdf": f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None
                }
            else:
                stop_paging = True
                break

        self.logger.info(f"第 {page} 页找到 {found_in_page} 条上周论文")

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
