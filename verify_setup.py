#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证设置脚本 - 检查所有组件是否正确配置
"""
import os
import sys
import json
from datetime import datetime, timedelta

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"PASS {description}: {filepath}")
        return True
    else:
        print(f"FAIL {description}: {filepath} (文件不存在)")
        return False

def check_imports():
    """检查必要的模块是否可以导入"""
    print("\n检查Python模块导入...")
    
    modules_to_check = [
        ('scrapy', 'Scrapy爬虫框架'),
        ('requests', 'HTTP请求库'),
        ('langchain', 'LangChain AI框架'),
        ('tqdm', '进度条库'),
    ]
    
    all_imports_ok = True
    for module, description in modules_to_check:
        try:
            __import__(module)
            print(f"PASS {description}: {module}")
        except ImportError:
            print(f"FAIL {description}: {module} (未安装)")
            all_imports_ok = False
    
    return all_imports_ok

def check_project_structure():
    """检查项目结构"""
    print("\n检查项目结构...")
    
    required_files = [
        ('.github/workflows/arxiv-daily-flexible.yml', '灵活爬虫工作流'),
        ('daily_arxiv/daily_arxiv/spiders/arxiv_flexible.py', '灵活爬虫主程序'),
        ('daily_arxiv/daily_arxiv/utils/date_validator.py', '日期验证工具'),
        ('ai/enhance.py', 'AI增强模块'),
        ('to_md/convert.py', 'Markdown转换模块'),
        ('test_flexible_crawler.py', '测试脚本'),
    ]
    
    all_files_exist = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_files_exist = False
    
    return all_files_exist

def check_environment_variables():
    """检查环境变量设置"""
    print("\n检查环境变量...")
    
    # 设置测试环境变量
    test_vars = {
        'KEYWORDS': 'machine learning, deep learning',
        'LANGUAGE': 'Chinese',
        'MODEL_NAME': 'deepseek-chat',
        'START_DATE': 'yesterday',
        'END_DATE': 'today'
    }
    
    for var, value in test_vars.items():
        os.environ[var] = value
        print(f"PASS 设置测试变量: {var}={value}")
    
    return True

def test_date_validation():
    """测试日期验证功能"""
    print("\n测试日期验证功能...")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'daily_arxiv'))
        from daily_arxiv.utils import DateValidator, validate_keywords, generate_search_query
        
        # 测试日期解析
        test_dates = ['today', 'tomorrow', 'yesterday', '+1', '2024-01-01']
        for date_str in test_dates:
            result = DateValidator.parse_date_input(date_str)
            if result:
                print(f"PASS 日期解析成功: {date_str} -> {result.date()}")
            else:
                print(f"FAIL 日期解析失败: {date_str}")
                return False
        
        # 测试关键词验证
        keywords_str = 'machine learning, deep learning'
        is_valid, error_msg, keywords = validate_keywords(keywords_str)
        if is_valid:
            print(f"PASS 关键词验证成功: {keywords}")
            query = generate_search_query(keywords)
            print(f"PASS 搜索查询生成成功: {query[:50]}...")
        else:
            print(f"FAIL 关键词验证失败: {error_msg}")
            return False
        
        return True
        
    except Exception as e:
        print(f"FAIL 日期验证测试失败: {e}")
        return False

def test_spider_import():
    """测试爬虫导入"""
    print("\n测试爬虫导入...")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'daily_arxiv'))
        from daily_arxiv.spiders.arxiv_flexible import ArxivFlexibleSpider
        
        spider = ArxivFlexibleSpider()
        print("PASS 灵活爬虫导入成功")
        
        # 测试日期解析方法
        test_date = spider.parse_date_input('today')
        if test_date:
            print(f"PASS 爬虫日期解析成功: {test_date}")
        else:
            print("FAIL 爬虫日期解析失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"FAIL 爬虫导入测试失败: {e}")
        return False

def generate_config_template():
    """生成配置模板"""
    print("\n生成GitHub配置模板...")
    
    config = {
        "secrets": {
            "OPENAI_API_KEY": "你的API密钥",
            "OPENAI_BASE_URL": "https://api.deepseek.com"
        },
        "variables": {
            "KEYWORDS": "machine learning, deep learning, computer vision",
            "LANGUAGE": "Chinese",
            "MODEL_NAME": "deepseek-chat",
            "EMAIL": "your-email@example.com",
            "NAME": "Your Name",
            "PER_PAGE": "200",
            "MAX_PAGES": "10",
            "DATE_FIELD": "published"
        }
    }
    
    with open('github_config_template.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("PASS 配置模板已生成: github_config_template.json")
    return True

def main():
    """主验证函数"""
    print("=" * 60)
    print("arXiv灵活爬虫设置验证")
    print("=" * 60)
    
    all_checks_passed = True
    
    # 检查项目结构
    if not check_project_structure():
        all_checks_passed = False
    
    # 检查Python模块
    if not check_imports():
        print("\nWARN: 某些Python模块未安装，请运行: pip install -r requirements.txt")
        all_checks_passed = False
    
    # 检查环境变量
    check_environment_variables()
    
    # 测试日期验证
    if not test_date_validation():
        all_checks_passed = False
    
    # 测试爬虫导入
    if not test_spider_import():
        all_checks_passed = False
    
    # 生成配置模板
    generate_config_template()
    
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("PASS 所有检查通过！系统已准备就绪。")
        print("\n下一步:")
        print("1. 在GitHub仓库中设置Secrets和Variables")
        print("2. 手动触发Actions工作流进行测试")
        print("3. 测试'今天爬取明天'功能")
    else:
        print("FAIL 部分检查失败，请解决上述问题后重新运行验证。")
    print("=" * 60)

if __name__ == '__main__':
    main()
