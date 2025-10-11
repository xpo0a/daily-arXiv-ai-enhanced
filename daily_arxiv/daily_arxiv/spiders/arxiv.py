# -*- coding: utf-8 -*-
import os
import scrapy
from datetime import datetime, timedelta
from urllib.parse import urlencode

class ArxivSpider(scrapy.Spider):
    name = "arxiv"

    def start_requests(self):
        """
        构建查询并从 arXiv API 分页拉取结果。
        配置项（可通过环境变量调整）:
          - KEYWORDS: 必需，逗号分隔关键词
          - TARGET_DATE: 可选，格式 YYYY-MM-DD（默认 UTC 昨日）
          - DATE_FIELD: 可选 'published' 或 'updated'（默认 'published'）
          - PER_PAGE: 每页数量（默认 200）
          - MAX_PAGES: 最多翻页次数防止无限循环（默认 10）
        """
        keywords_str = os.environ.get('KEYWORDS')
        if not keywords_str:
            self.logger.error("错误：未设置 KEYWORDS 环境变量（例如：'Robotics,Model Quantization'）")
            return

        # 清理关键词，去掉可能的多余引号
        keywords = [k.strip().strip('"').strip("'") for k in keywords_str.split(',') if k.strip()]
        if not keywords:
            self.logger.error("错误：KEYWORDS 为空或格式不正确。")
            return

        self.logger.info(f"成功加载关键词: {keywords}")

        # 构建 search_query（每个关键词在标题或摘要中的短语匹配）
        # 举例: (ti:"Robotics" OR abs:"Robotics") OR (ti:"Model Quantization" OR abs:"Model Quantization")
        query_parts = [f'(ti:"{k}" OR abs:"{k}")' for k in keywords]
        self.search_query = ' OR '.join(query_parts)

        # 参数
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
            # 默认目标为 UTC 昨日（和你之前一致）
            self.target_date = (datetime.utcnow().date() - timedelta(days=1))

        self.logger.info(f"将只保留 {self.date_field} 于 {self.target_date} (UTC) 的论文")
        # 初始请求 start=0
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
        关键逻辑：
         - 按 page 顺序处理条目（arXiv 返回的是 Atom）
         - 如果条目 date > target_date：继续（还未到目标日）
         - 如果 == target_date：yield item
         - 如果 < target_date：本页之后均更早 -> 停止翻页（safe stop）
        """
        start = response.meta.get('start', 0)
        page = response.meta.get('page', 1)

        # 使用不依赖 namespace 的 XPath（local-name()），避免命名空间麻烦
        entries = response.xpath('//*[local-name()="entry"]')
        if not entries:
            self.logger.warning(f"第 {page} 页未返回 entry，可能 API 响应结构改变或关键词无匹配。")
            return

        found_in_page = 0
        stop_paging = False
        # 用于调试：记录本页见到的前若干个日期
        page_dates = []

        for entry in entries:
            date_text = entry.xpath(f'*[local-name()="{self.date_field}"]/text()').get()
            if not date_text:
                # 没有对应日期字段，跳过该条
                self.logger.debug("条目缺少日期字段，跳过。")
                continue

            # 兼容 Z 时区
            try:
                pub_dt = datetime.fromisoformat(date_text.replace('Z', '+00:00'))
            except Exception:
                # 容错：有时可能是其他格式，尽量解析失败时继续
                self.logger.debug(f"无法解析时间：{date_text}，跳过该条目。")
                continue

            pub_date = pub_dt.date()
            page_dates.append(pub_date.isoformat())

            # 条目比目标更新/提交更晚 -> 继续（因为我们按降序）
            if pub_date > self.target_date:
                continue
            # 等于目标 -> 输出
            elif pub_date == self.target_date:
                found_in_page += 1
                arxiv_id_url = entry.xpath('*[local-name()="id"]/text()').get() or ''
                arxiv_id = arxiv_id_url.split('/')[-1] if arxiv_id_url else None
                title = entry.xpath('*[local-name()="title"]/text()').get()
                summary = entry.xpath('*[local-name()="summary"]/text()').get()
                authors = entry.xpath('*[local-name()="author"]/*[local-name()="name"]/text()').getall()
                # 构造 pdf 链接（安全方式）
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
                # 遇到更早的日期，说明目标日的条目在此之前已经结束，可以停止翻页
                stop_paging = True
                break

        self.logger.debug(f"第 {page} 页找到 {found_in_page} 条目标日期论文；seen_dates（部分）={page_dates[:8]}")

        # 如果本页没有命中目标并且本页最早条目比目标还新，则说明还需继续下一页
        if not stop_paging and len(entries) == self.per_page:
            next_start = start + self.per_page
            next_page = page + 1
            if next_page > self.max_pages:
                self.logger.warning(f"已到达 max_pages ({self.max_pages})，停止翻页以避免无限循环。")
                # 若一页都没命中并且可能在更后面，建议改用 OAI-PMH 或增大 MAX_PAGES
                if found_in_page == 0:
                    self.logger.info("未找到目标日期论文 —— 考虑使用 OAI-PMH 全量按日抓取以保证不漏稿。")
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
                # 本页条目数 < per_page，说明已到数据尾部
                self.logger.info("已到达结果末尾，停止翻页。")

            # 如果到头了但从未找到任何目标日论文，打印部分调试信息，便于排查
            if found_in_page == 0:
                self.logger.warning(f"在已请求的结果中未找到 {self.target_date} 的论文。样例本页日期（最多 8 个）: {page_dates[:8]}")
                self.logger.info("如果你确信当天应有匹配，建议：1) 增大 MAX_PAGES 或 PER_PAGE；2) 使用 OAI-PMH 按日抓全量后再本地关键词过滤。")
