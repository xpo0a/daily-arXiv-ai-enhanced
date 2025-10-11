#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试灵活爬虫功能
"""
import os
import sys
import tempfile
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'daily_arxiv'))

from daily_arxiv.utils import DateValidator, validate_keywords, generate_search_query


def test_date_validation():
    """测试日期验证功能"""
    print("测试日期验证功能...")
    
    # 测试相对日期
    test_cases = [
        ('today', True),
        ('tomorrow', True),
        ('yesterday', True),
        ('+1', True),
        ('-1', True),
        ('+7', True),
        ('2024-01-01', True),
        ('2024/01/01', True),
        ('invalid-date', False),
        ('', False),
    ]
    
    for date_str, should_pass in test_cases:
        result = DateValidator.parse_date_input(date_str)
        status = "PASS" if (result is not None) == should_pass else "FAIL"
        print(f"  {status} {date_str}: {result}")
    
    # 测试日期范围验证
    print("\n测试日期范围验证...")
    range_cases = [
        ('today', 'tomorrow', True),
        ('yesterday', 'today', True),
        ('+1', '+7', True),
        ('2024-01-01', '2024-01-02', True),
        ('tomorrow', 'yesterday', True),  # 应该自动交换
        ('2024-01-01', '2025-01-01', False),  # 超过365天
    ]
    
    for start, end, should_pass in range_cases:
        is_valid, error_msg, parsed_start, parsed_end = DateValidator.validate_date_range(start, end)
        status = "PASS" if is_valid == should_pass else "FAIL"
        print(f"  {status} {start} 到 {end}: {is_valid} - {error_msg}")


def test_keyword_validation():
    """测试关键词验证功能"""
    print("\n测试关键词验证功能...")
    
    test_cases = [
        ('machine learning, deep learning', True),
        ('AI, computer vision, NLP', True),
        ('', False),
        ('a', False),  # 太短
        ('x' * 101, False),  # 太长
        (','.join([f'keyword{i}' for i in range(25)]), False),  # 太多关键词
    ]
    
    for keywords_str, should_pass in test_cases:
        is_valid, error_msg, keywords = validate_keywords(keywords_str)
        status = "PASS" if is_valid == should_pass else "FAIL"
        print(f"  {status} '{keywords_str}': {is_valid} - {error_msg}")
        if is_valid:
            query = generate_search_query(keywords)
            print(f"    生成的查询: {query[:100]}...")


def test_crawler_integration():
    """测试爬虫集成"""
    print("\n测试爬虫集成...")
    
    # 设置测试环境变量
    os.environ['KEYWORDS'] = 'machine learning, deep learning'
    os.environ['START_DATE'] = 'yesterday'
    os.environ['END_DATE'] = 'today'
    os.environ['LANGUAGE'] = 'Chinese'
    os.environ['MODEL_NAME'] = 'deepseek-chat'
    
    try:
        from daily_arxiv.spiders.arxiv_flexible import ArxivFlexibleSpider
        
        spider = ArxivFlexibleSpider()
        
        # 测试日期解析
        start_date = spider.parse_date_input('yesterday')
        end_date = spider.parse_date_input('today')
        
        print(f"  PASS 日期解析成功: {start_date} 到 {end_date}")
        
        # 测试关键词解析
        keywords_str = os.environ.get('KEYWORDS', '')
        is_valid, error_msg, keywords = validate_keywords(keywords_str)
        
        if is_valid:
            search_query = generate_search_query(keywords)
            print(f"  PASS 关键词解析成功: {keywords}")
            print(f"  PASS 搜索查询: {search_query}")
        else:
            print(f"  FAIL 关键词解析失败: {error_msg}")
            
    except ImportError as e:
        print(f"  WARN 无法导入爬虫模块: {e}")
    except Exception as e:
        print(f"  FAIL 测试失败: {e}")


def main():
    """主测试函数"""
    print("开始测试灵活爬虫功能\n")
    
    test_date_validation()
    test_keyword_validation()
    test_crawler_integration()
    
    print("\n测试完成！")
    print("\n使用说明:")
    print("1. 在GitHub仓库的Settings -> Secrets and variables -> Actions中设置:")
    print("   - KEYWORDS: 用逗号分隔的关键词，如 'machine learning, deep learning'")
    print("   - LANGUAGE: 语言，如 'Chinese' 或 'English'")
    print("   - MODEL_NAME: 模型名称，如 'deepseek-chat'")
    print("   - OPENAI_API_KEY: API密钥")
    print("   - OPENAI_BASE_URL: API基础URL")
    print("\n2. 手动触发工作流时可以使用以下日期格式:")
    print("   - 相对日期: today, tomorrow, yesterday, +1, -1")
    print("   - 绝对日期: 2024-01-01, 2024/01/01")
    print("\n3. 工作流文件: .github/workflows/arxiv-daily-flexible.yml")


if __name__ == '__main__':
    main()
