# -*- coding: utf-8 -*-
import os
import time
import scrapy
import requests
from datetime import datetime, timedelta
from urllib.parse import urlencode


class ArxivSpider(scrapy.Spider):
    name = "arxiv"

    def wait_until_data_ready(self):
        """
        等待 arXiv 昨日数据更新完成。
        原理：通过请求最新提交，判断是否包含昨日的论文。
        """
        check_interval = int(os.environ.get("CHECK_INTERVAL", 1800))  # 默认30分钟
        max_retry = int(os.environ.get("MAX_RETRY", 12))  # 最多重试12次（6小时）
        base_url = "https://export.arxiv.org/api/query?"

        for attempt in range(1, max_retry + 1):
            if self.is_yesterday_data_ready(base_url):
                self.logger.info("✅ 昨日数据已更新，开始爬取。")
                return True
            else:
                self.logger.warning(f"⚠️ 第 {attempt} 次检测：昨日数据尚未更新，{check_interval/60:.0f} 分钟后重试...")
                time.sleep(check_interval)

        self.logger.error("❌ 超过最大重试次数，昨日数据仍未更新。终止爬取。")
        return False

    def is_yesterday_data_ready(self, base_url: str) -> bool:
        """
        检查昨日数据是否可用。
        逻辑：请求最新论文，解析 <published> 字段，如果有等于昨日的日期则认为更新完成。
        """
        yesterday = (datetime.utcnow().date() - timedelta(days=1))
        params = {
            "search_query": "all",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": 10,
        }
    
        try:
            url = base_url + urlencode(params)
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
    
            # 用 XPath 解析（不依赖命名空间）
            import lxml.etree as ET
            root = ET.fromstring(resp.text.encode("utf-8"))
            dates = [
                datetime.fromisoformat(d.replace("Z", "+00:00")).date()
                for d in root.xpath('//*[local-name()="published"]/text()')
                if d
            ]
            if not dates:
                self.logger.warning("检测时未解析到 published 日期。")
                return False
    
            max_date = max(dates)
            self.logger.info(f"检测到最新 published 日期为 {max_date}（样例: {dates[:5]}）")
            if yesterday in dates:
                self.logger.info(f"✅ 确认 {yesterday} 已出现在 published 中，数据已更新。")
                return True
            else:
                self.logger.info(f"⚠️ 最新 published={max_date}，仍早于昨日 {yesterday}。")
                return False
    
        except Exception as e:
            self.logger.warning(f"检测昨日数据失败：{e}")
            return False


    def start_requests(self):
        """
        构建查询并从 arXiv API 分页拉取结果。
        配置项（可通过环境变量调整）:
          - KEYWORDS: 必需，逗号分隔关键词
          - TARGET_DATE: 可选，格式 YYYY-MM-DD（默认 UTC 昨日）
          - DATE_FIELD: 可选 'published' 或 'updated'（默认 'published'）
          - PER_PAGE: 每页数量（默认 200）
          - MAX_PAGES: 最多翻页次数防止无限循环（默认 10）
          - CHECK_INTERVAL: 检测间隔秒数（默认1800）
          - MAX_RETRY: 最大重试次数（默认12）
        """

        # === 等待数据更新完成 ===
        if not self.wait_until_data_ready():
            return  # 放弃继续执行

        # === 原始逻辑 ===
        keywords_str = os.environ.get('KEYWORDS')
        if not keywords_str:
            self.logger.error("错误：未设置 KEYWORDS 环境变量（例如：'Robotics,Model Quantization'）")
            return

        # 清理关键词
        keywords = [k.strip().strip('"').strip("'") for k in keywords_str.split(',') if k.strip()]
        if not keywords:
            self.logger.error("错误：KEYWORDS 为空或格式不正确。")
            return

        self.logger.info(f"成功加载关键词: {keywords}")

        # 构建 search_query
        query_parts = [f'(ti:"{k}" OR abs:"{k}")' for k in keywords]
        self.search_query = ' OR '.join(query_parts)

        # 参数初始化
        self.base_url = "https://export.arxiv.org/api/query?"
        self.per_page = int(os.environ.get('PER_PAGE', 200))
        self.max_pages = int(os.environ.get('MAX_PAGES', 10))
        self.date_field = os.environ.get('DATE_FIELD', 'published').strip().lower()
        if self.date_field not in ('published', 'updated'):
            self.logger.warning("DATE_FIELD 非预期，改回 'published'")
            self.date_field = 'published'

        target_date_env = os.environ.get('TARGET_DATE')
        if target_date_env:
            try:
                self.target_date = datetime.strptime(target_date_env, "%Y-%m-%d").date()
            except Exception:
                self.logger.error("TARGET_DATE 格式错误，应为 YYYY-MM-DD。使用 UTC 昨日代替。")
                self.target_date = (datetime.utcnow().date() - timedelta(days=1))
        else:
            self.target_date = (datetime.utcnow().date() - timedelta(days=1))

        self.logger.info(f"将只保留 {self.date_field} 于 {self.target_date} (UTC) 的论文")

        # 起始请求
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
        解析单页 API 返回并按需翻页。
        """
        start = response.meta.get('start', 0)
        page = response.meta.get('page', 1)

        entries = response.xpath('//*[local-name()="entry"]')
        if not entries:
            self.logger.warning(f"第 {page} 页未返回 entry，可能 API 响应结构改变或关键词无匹配。")
            return

        found_in_page = 0
        stop_paging = False
        page_dates = []

        for entry in entries:
            date_text = entry.xpath(f'*[local-name()="{self.date_field}"]/text()').get()
            if not date_text:
                continue
            try:
                pub_dt = datetime.fromisoformat(date_text.replace('Z', '+00:00'))
            except Exception:
                continue

            pub_date = pub_dt.date()
            page_dates.append(pub_date.isoformat())

            if pub_date > self.target_date:
                continue
            elif pub_date == self.target_date:
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
                stop_paging = True
                break

        self.logger.debug(f"第 {page} 页找到 {found_in_page} 条目标日期论文；seen_dates（部分）={page_dates[:8]}")

        if not stop_paging and len(entries) == self.per_page:
            next_start = start + self.per_page
            next_page = page + 1
            if next_page > self.max_pages:
                self.logger.warning(f"已到达 max_pages ({self.max_pages})，停止翻页。")
                if found_in_page == 0:
                    self.logger.info("未找到目标日期论文 —— 可增大 MAX_PAGES 或使用 OAI-PMH。")
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
                self.logger.info(f"已遇到早于 {self.target_date} 的条目，停止翻页。")
            else:
                self.logger.info("已到达结果末尾，停止翻页。")

            if found_in_page == 0:
                self.logger.warning(f"未找到 {self.target_date} 的论文，样例日期: {page_dates[:8]}")
