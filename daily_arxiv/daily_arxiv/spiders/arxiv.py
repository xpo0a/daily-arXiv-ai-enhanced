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
        """
        等待 arXiv 上周数据更新完成。
        检查逻辑：检测上周日的数据是否已可用。
        """
        check_interval = int(os.environ.get("CHECK_INTERVAL", 1800))  # 默认30分钟
        max_retry = int(os.environ.get("MAX_RETRY", 12))  # 默认6小时内尝试12次
        base_url = "https://export.arxiv.org/api/query?"

        for attempt in range(1, max_retry + 1):
            if self.is_last_week_data_ready(base_url):
                self.logger.info("✅ 上周数据已更新，开始爬取。")
                return True
            else:
                self.logger.warning(f"⚠️ 第 {attempt} 次检测：上周数据尚未更新，{check_interval/60:.0f} 分钟后重试...")
                time.sleep(check_interval)

        self.logger.error("❌ 超过最大重试次数，上周数据仍未更新。终止爬取。")
        return False

    def is_last_week_data_ready(self, base_url: str) -> bool:
        """
        检查上周数据是否更新完毕（上周日的数据是否可用）。
        """
        last_sunday = (datetime.utcnow().date() - timedelta(days=datetime.utcnow().weekday() + 1))
        params = {
            "search_query": "all",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": 5,
        }
        try:
            url = base_url + urlencode(params)
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            if str(last_sunday) in resp.text:
                self.logger.info(f"检测到上周日 ({last_sunday}) 出现在最新结果中，数据已更新。")
                return True
            return False
        except Exception as e:
            self.logger.warning(f"检测上周数据失败：{e}")
            return False

    def start_requests(self):
        """
        每周一运行，自动爬取上周一～上周日的数据。
        """
        # === 等待上周数据更新 ===
        if not self.wait_until_data_ready():
            return  # 数据未准备好则退出

        # === 获取关键词 ===
        keywords_str = os.environ.get('KEYWORDS')
        if not keywords_str:
            self.logger.error("错误：未设置 KEYWORDS 环境变量（例如：'Robotics,Model Quantization'）")
            return

        keywords = [k.strip().strip('"').strip("'") for k in keywords_str.split(',') if k.strip()]
        if not keywords:
            self.logger.error("错误：KEYWORDS 为空或格式不正确。")
            return

        self.logger.info(f"成功加载关键词: {keywords}")

        # === 构建搜索查询 ===
        query_parts = [f'(ti:"{k}" OR abs:"{k}")' for k in keywords]
        self.search_query = ' OR '.join(query_parts)

        # === 设置上周时间范围 ===
        today = datetime.utcnow().date()
        self.start_date = today - timedelta(days=today.weekday() + 7)   # 上周一
        self.end_date = self.start_date + timedelta(days=6)             # 上周日
        self.logger.info(f"📅 将爬取上周 ({self.start_date} ~ {self.end_date}) 的论文")

        # === 参数初始化 ===
        self.base_url = "https://export.arxiv.org/api/query?"
        self.per_page = int(os.environ.get('PER_PAGE', 200))
        self.max_pages = int(os.environ.get('MAX_PAGES', 10))
        self.date_field = os.environ.get('DATE_FIELD', 'published').strip().lower()
        if self.date_field not in ('published', 'updated'):
            self.logger.warning("DATE_FIELD 非预期，改回 'published'")
            self.date_field = 'published'

        # === 起始请求 ===
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
        """
        解析每页的 API 返回，过滤出上周的论文。
        """
        start = response.meta.get('start', 0)
        page = response.meta.get('page', 1)

        entries = response.xpath('//*[local-name()="entry"]')
        if not entries:
            self.logger.warning(f"第 {page} 页未返回 entry，可能 API 响应结构改变或关键词无匹配。")
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
                continue  # 仍是本周或未来数据
            elif self.start_date <= pub_date <= self.end_date:
                found_in_page += 1
                arxiv_id_url = entry.xpath('*[local-name()="id"]/text()').get() or ''
                arxiv_id = arxiv_id_url.split('/')[-1] if arxiv_id_url else None
                title = entry.xpath('*[local-name()="title"]/text()').get()
                summary = entry.xpath('*[local-name()="summary"]/text()').get()
                authors = entry.xpath('*[local-name()="author"]/*[local-name()="name"]/text()').getall()
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None

                yield {
                    "id": arxiv_id,
                    "title": title.strip() if title else None,
                    "summary": summary.strip() if summary else None,
                    "authors": authors,
                    self.date_field: date_text,
                    "pdf": pdf_url,
                }
            else:
                # 早于上周数据，停止翻页
                stop_paging = True
                break

        self.logger.info(f"第 {page} 页找到 {found_in_page} 条上周论文")

        # 翻页逻辑
        if not stop_paging and len(entries) == self.per_page:
            next_start = start + self.per_page
            next_page = page + 1
            if next_page > self.max_pages:
                self.logger.warning(f"已到达 max_pages ({self.max_pages})，停止翻页。")
                return

            params = {
                'search_query': self.search_query,
                'start': next_start,
                'max_results': self.per_page,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            next_url = self.base_url + urlencode(params)
            self.logger.info(f"继续翻页：请求 start={next_start} (page {next_page})")
            yield scrapy.Request(next_url, callback=self.parse_api_response, meta={'start': next_start, 'page': next_page})
        else:
            if stop_paging:
                self.logger.info(f"📚 已遇到早于 {self.start_date} 的条目，停止翻页。")
            else:
                self.logger.info("🛑 已到达结果末尾，停止翻页。")
