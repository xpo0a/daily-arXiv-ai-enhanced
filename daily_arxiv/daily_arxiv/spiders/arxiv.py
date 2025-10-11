# -*- coding: utf-8 -*-
import scrapy
import os
import re
from datetime import datetime, timedelta
from urllib.parse import quote

class ArxivSpider(scrapy.Spider):
    name = "arxiv"  # 爬虫名称
    
    def start_requests(self):
        """
        这个方法将替代旧的初始化方法。
        它会读取环境变量中的KEYWORDS，并构建一个合法的arXiv API搜索请求。
        """
        # 从环境变量中读取关键词字符串
        keywords_str = os.environ.get('KEYWORDS')
        if not keywords_str:
            self.logger.error("错误：未在仓库变量中设置 KEYWORDS。请添加您感兴趣的关键词。")
            return

        # 将逗号分隔的字符串转换为关键词列表
        keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
        if not keywords:
            self.logger.error("错误：KEYWORDS 变量为空或格式不正确。")
            return

        self.logger.info(f"成功加载关键词: {keywords}")

        # 构建API搜索查询
        # 格式为: (ti:"keyword1" OR abs:"keyword1") OR (ti:"keyword2" OR abs:"keyword2")
        # ti=标题, abs=摘要
        query_parts = []
        for keyword in keywords:
            # 对包含空格的关键词进行URL编码和引号处理，以实现精确短语搜索
            encoded_keyword = quote(f'"{keyword}"')
            query_parts.append(f'(ti:{encoded_keyword} OR abs:{encoded_keyword})')

        # 使用 "+OR+" 连接不同的关键词查询
        search_query = '+OR+'.join(query_parts)

        # 构建最终的API请求URL
        # sortBy=submittedDate&sortOrder=descending 获取最新提交的论文
        # max_results=200 获取足够多的近期论文，后续再进行日期过滤
        base_url = "http://export.arxiv.org/api/query?"
        api_url = f"{base_url}search_query={search_query}&sortBy=submittedDate&sortOrder=descending&max_results=200"

        self.logger.info(f"构造的API请求URL: {api_url}")
        
        # 发送请求
        yield scrapy.Request(url=api_url, callback=self.parse_api_response)

    def parse_api_response(self, response):
        """
        这个方法用于解析arXiv API返回的XML数据。
        它替代了旧的、用于解析HTML的parse方法。
        """
        # 注册Atom XML的命名空间，以便XPath可以正确工作
        response.selector.register_namespace("atom", "http://www.w3.org/2005/Atom")

        # 提取所有论文条目
        entries = response.xpath("//atom:entry")
        
        if not entries:
            self.logger.warning("API未返回任何论文条目，请检查关键词或arXiv API状态。")
            return

        # 获取昨天的日期（UTC），因为arXiv的"new"通常指的是前一个工作日的论文
        utc_yesterday = datetime.utcnow().date() - timedelta(days=1)
        self.logger.info(f"将只保留发布于 {utc_yesterday} (UTC) 的论文")

        found_papers_count = 0
        for entry in entries:
            # 提取论文的发布日期
            published_date_str = entry.xpath("atom:published/text()").get()
            published_date = datetime.fromisoformat(published_date_str.replace("Z", "+00:00")).date()

            # 只处理昨天的论文
            if published_date != utc_yesterday:
                # 由于结果是按日期排序的，一旦日期不匹配，可以提前停止
                if found_papers_count > 0:
                    self.logger.info(f"已处理完所有 {utc_yesterday} 的论文，停止解析。")
                    break
                continue

            found_papers_count += 1
            
            # 提取论文ID
            arxiv_id_url = entry.xpath("atom:id/text()").get()
            arxiv_id = arxiv_id_url.split('/')[-1]

            yield {
                "id": arxiv_id,
                # 你可以从这里提取更多信息，并传递给后续的pipeline
                # "title": entry.xpath("atom:title/text()").get().strip(),
                # "summary": entry.xpath("atom:summary/text()").get().strip(),
            }
        
        if found_papers_count == 0:
            self.logger.warning(f"在API返回的最新200篇论文中，没有找到发布于 {utc_yesterday} (UTC) 的论文。")
