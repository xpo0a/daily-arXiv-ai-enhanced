#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试OR逻辑的搜索查询生成
"""
import sys
import os
sys.path.append('daily_arxiv')

from daily_arxiv.utils.date_validator import generate_search_query, validate_keywords

def test_or_logic():
    """测试OR逻辑的搜索查询"""
    
    # 你的关键词
    keywords_str = "Embodied AI, Robot Learning, Robotics, Model Quantization, Quantization, Model Compression, Network Pruning, Model Acceleration, Efficient Inference, Large Multimodal Model, LMM, Vision-Language, Vision and Language, Multimodal"
    
    print("Testing OR logic search query")
    print("=" * 50)
    print(f"Original keywords: {keywords_str}")
    print()
    
    # 验证关键词
    is_valid, error_msg, keywords = validate_keywords(keywords_str)
    if not is_valid:
        print(f"Keywords validation failed: {error_msg}")
        return
    
    print(f"Keywords validation passed, total {len(keywords)} keywords:")
    for i, keyword in enumerate(keywords, 1):
        print(f"  {i:2d}. {keyword}")
    print()
    
    # 生成搜索查询
    search_query = generate_search_query(keywords)
    print("Generated search query (OR logic):")
    print("-" * 50)
    print(search_query)
    print()
    
    # 分析查询结构
    print("Query analysis:")
    print(f"  - Number of keywords: {len(keywords)}")
    print(f"  - Query length: {len(search_query)} characters")
    print(f"  - Number of OR connectors: {search_query.count(' OR ')}")
    print()
    
    # 显示前几个关键词的查询部分
    print("First 5 keywords query parts:")
    for i, keyword in enumerate(keywords[:5], 1):
        escaped_keyword = keyword.replace('"', '\\"')
        query_part = f'(ti:"{escaped_keyword}" OR abs:"{escaped_keyword}")'
        print(f"  {i}. {query_part}")
    print()
    
    print("OR logic test completed!")
    print("Now any paper containing any of the keywords will be matched")

if __name__ == "__main__":
    test_or_logic()
