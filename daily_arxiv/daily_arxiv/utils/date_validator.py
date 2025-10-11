# -*- coding: utf-8 -*-
"""
日期验证和解析工具
"""
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple


class DateValidator:
    """日期验证和解析类"""
    
    @staticmethod
    def parse_date_input(date_str: str) -> Optional[datetime]:
        """
        解析日期输入，支持多种格式
        
        Args:
            date_str: 日期字符串，支持以下格式：
                - 'today', 'tomorrow', 'yesterday'
                - '+N' 或 '-N' (相对天数)
                - 'YYYY-MM-DD' 或 'YYYY/MM/DD' (绝对日期)
        
        Returns:
            datetime对象，如果解析失败返回None
        """
        if not date_str:
            return None
        
        date_str = date_str.strip().lower()
        
        # 相对日期
        if date_str == 'today':
            return datetime.utcnow()
        elif date_str == 'tomorrow':
            return datetime.utcnow() + timedelta(days=1)
        elif date_str == 'yesterday':
            return datetime.utcnow() - timedelta(days=1)
        elif date_str == 'last_monday':
            # 计算上一周的周一
            today = datetime.utcnow().date()
            days_since_monday = today.weekday()  # 0=Monday, 6=Sunday
            # 上一周的周一 = 今天 - 今天到本周一的天数 - 7天
            last_monday = today - timedelta(days=days_since_monday + 7)
            return datetime.combine(last_monday, datetime.min.time())
        elif date_str == 'last_sunday':
            # 计算上一周的周日
            today = datetime.utcnow().date()
            days_since_monday = today.weekday()  # 0=Monday, 6=Sunday
            # 上一周的周日 = 今天 - 今天到本周一的天数 - 1天
            last_sunday = today - timedelta(days=days_since_monday + 1)
            return datetime.combine(last_sunday, datetime.min.time())
        elif re.match(r'^[+-]\d+$', date_str):
            # 相对天数，如 +1, -2
            try:
                days = int(date_str)
                return datetime.utcnow() + timedelta(days=days)
            except ValueError:
                return None
        else:
            # 绝对日期格式
            formats = ['%Y-%m-%d', '%Y/%m/%d', '%Y-%m-%d %H:%M:%S']
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            return None
    
    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> Tuple[bool, str, Optional[datetime], Optional[datetime]]:
        """
        验证日期范围
        
        Args:
            start_date: 开始日期字符串
            end_date: 结束日期字符串
        
        Returns:
            (is_valid, error_message, parsed_start, parsed_end)
        """
        parsed_start = DateValidator.parse_date_input(start_date)
        parsed_end = DateValidator.parse_date_input(end_date)
        
        if not parsed_start:
            return False, f"无效的开始日期格式: {start_date}", None, None
        
        if not parsed_end:
            return False, f"无效的结束日期格式: {end_date}", None, None
        
        # 检查日期范围是否合理（不能超过1年）
        if (parsed_end - parsed_start).days > 365:
            return False, "日期范围不能超过365天", parsed_start, parsed_end
        
        # 检查开始日期不能晚于结束日期超过30天
        if (parsed_start - parsed_end).days > 30:
            return False, "开始日期不能晚于结束日期超过30天", parsed_start, parsed_end
        
        return True, "", parsed_start, parsed_end
    
    @staticmethod
    def format_date_for_filename(date_obj: datetime) -> str:
        """
        将日期对象格式化为文件名格式
        
        Args:
            date_obj: 日期对象
        
        Returns:
            格式化的日期字符串 (YYYY-MM-DD)
        """
        return date_obj.strftime('%Y-%m-%d')
    
    @staticmethod
    def get_date_range_description(start_date: str, end_date: str) -> str:
        """
        获取日期范围的描述
        
        Args:
            start_date: 开始日期字符串
            end_date: 结束日期字符串
        
        Returns:
            日期范围描述
        """
        parsed_start = DateValidator.parse_date_input(start_date)
        parsed_end = DateValidator.parse_date_input(end_date)
        
        if not parsed_start or not parsed_end:
            return f"{start_date} 到 {end_date}"
        
        start_str = DateValidator.format_date_for_filename(parsed_start)
        end_str = DateValidator.format_date_for_filename(parsed_end)
        
        if start_str == end_str:
            return start_str
        else:
            return f"{start_str} 到 {end_str}"


def validate_keywords(keywords_str: str) -> Tuple[bool, str, list]:
    """
    验证关键词字符串
    
    Args:
        keywords_str: 关键词字符串，用逗号分隔
    
    Returns:
        (is_valid, error_message, keywords_list)
    """
    if not keywords_str:
        return False, "关键词不能为空", []
    
    keywords = [k.strip().strip('"').strip("'") for k in keywords_str.split(',') if k.strip()]
    
    if not keywords:
        return False, "没有找到有效的关键词", []
    
    # 检查关键词长度
    for keyword in keywords:
        if len(keyword) < 2:
            return False, f"关键词 '{keyword}' 太短，至少需要2个字符", []
        if len(keyword) > 100:
            return False, f"关键词 '{keyword}' 太长，最多100个字符", []
    
    # 检查关键词数量
    if len(keywords) > 20:
        return False, f"关键词数量过多，最多支持20个关键词，当前有{len(keywords)}个", []
    
    return True, "", keywords


def generate_search_query(keywords: list) -> str:
    """
    根据关键词列表生成arXiv搜索查询（使用OR逻辑）
    
    Args:
        keywords: 关键词列表
    
    Returns:
        格式化的搜索查询字符串
    """
    if not keywords:
        return ""
    
    # 为每个关键词创建 (ti:"keyword" OR abs:"keyword") 格式
    # 使用OR逻辑连接所有关键词，这样只要论文包含任意一个关键词就会被匹配
    query_parts = []
    for keyword in keywords:
        # 转义特殊字符
        escaped_keyword = keyword.replace('"', '\\"')
        query_parts.append(f'(ti:"{escaped_keyword}" OR abs:"{escaped_keyword}")')
    
    # 使用OR连接所有关键词，这样匹配更宽松
    return ' OR '.join(query_parts)
