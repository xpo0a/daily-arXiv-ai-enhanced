#!/usr/bin/env python3
"""
检查Scrapy爬取统计信息的脚本 / Script to check Scrapy crawling statistics
用于获取去重检查的状态结果 / Used to get deduplication check status results

功能说明 / Features:
- 检查当日与昨日论文数据的重复情况 / Check duplication between today's and yesterday's paper data
- 删除重复论文条目，保留新内容 / Remove duplicate papers, keep new content
- 根据去重后的结果决定工作流是否继续 / Decide workflow continuation based on deduplication results
"""
import json
import sys
import os
from datetime import datetime, timedelta

def load_papers_data(file_path):
    """
    从jsonl文件中加载完整的论文数据
    Load complete paper data from jsonl file
    
    Args:
        file_path (str): JSONL文件路径 / JSONL file path
        
    Returns:
        list: 论文数据列表 / List of paper data
        set: 论文ID集合 / Set of paper IDs
    """
    if not os.path.exists(file_path):
        return [], set()
    
    papers = []
    ids = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    papers.append(data)
                    ids.add(data.get('id', ''))
        return papers, ids
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return [], set()

def save_papers_data(papers, file_path):
    """
    保存论文数据到jsonl文件
    Save paper data to jsonl file
    
    Args:
        papers (list): 论文数据列表 / List of paper data
        file_path (str): 文件路径 / File path
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for paper in papers:
                f.write(json.dumps(paper, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        print(f"Error saving {file_path}: {e}", file=sys.stderr)
        return False

def perform_deduplication():
    """
    执行多日去重：删除与历史多日重复的论文条目，保留新内容
    Perform deduplication over multiple past days
    
    Returns:
        str: 去重状态 / Deduplication status
             - "has_new_content": 有新内容 / Has new content
             - "no_new_content": 无新内容 / No new content  
             - "no_data": 无数据 / No data
             - "error": 处理错误 / Processing error
    """

    # 支持灵活的文件名检测
    today = datetime.now().strftime("%Y-%m-%d")
    today_file = f"../data/{today}.jsonl"
    
    # 如果今日文件不存在，尝试查找最新的数据文件
    if not os.path.exists(today_file):
        print("Today's data file does not exist, looking for latest data file...", file=sys.stderr)
        
        # 查找data目录下最新的jsonl文件
        data_dir = "../data"
        if os.path.exists(data_dir):
            jsonl_files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl') and not f.endswith('_AI_enhanced_')]
            if jsonl_files:
                # 按修改时间排序，获取最新的文件
                jsonl_files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
                latest_file = jsonl_files[0]
                today_file = f"../data/{latest_file}"
                print(f"Found latest data file: {latest_file}", file=sys.stderr)
            else:
                print("No data files found", file=sys.stderr)
                return "no_data"
        else:
            print("Data directory does not exist", file=sys.stderr)
            return "no_data"

    if not os.path.exists(today_file):
        print("Data file does not exist", file=sys.stderr)
        return "no_data"

    try:
        today_papers, today_ids = load_papers_data(today_file)
        print(f"Total papers: {len(today_papers)}", file=sys.stderr)

        if not today_papers:
            return "no_data"

        # 收集历史多日 ID 集合
        history_ids = set()
        for i in range(1, history_days + 1):
            date_str = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            history_file = f"../data/{date_str}.jsonl"
            _, past_ids = load_papers_data(history_file)
            history_ids.update(past_ids)

        print(f"History {history_days} days deduplication library size: {len(history_ids)}", file=sys.stderr)

        duplicate_ids = today_ids & history_ids

        if duplicate_ids:
            print(f"Found {len(duplicate_ids)} historical duplicate papers", file=sys.stderr)
            new_papers = [paper for paper in today_papers if paper.get('id', '') not in duplicate_ids]

            print(f"Remaining papers after deduplication: {len(new_papers)}", file=sys.stderr)

            if new_papers:
                if save_papers_data(new_papers, today_file):
                    print(f"Updated file, removed {len(duplicate_ids)} duplicate papers", file=sys.stderr)
                    return "has_new_content"
                else:
                    print("Failed to save deduplicated data", file=sys.stderr)
                    return "error"
            else:
                try:
                    os.remove(today_file)
                    print("All papers are duplicate content, file deleted", file=sys.stderr)
                except Exception as e:
                    print(f"Failed to delete file: {e}", file=sys.stderr)
                return "no_new_content"
        else:
            print("All content is new", file=sys.stderr)
            return "has_new_content"

    except Exception as e:
        print(f"Deduplication processing failed: {e}", file=sys.stderr)
        return "error"

def main():
    """
    Check deduplication status and return corresponding exit code
    
    Exit code meanings:
    0: Has new content, continue processing
    1: No new content, stop workflow
    2: Processing error
    """
    
    print("Performing intelligent deduplication check...", file=sys.stderr)
    
    # Perform deduplication processing
    dedup_status = perform_deduplication()
    
    if dedup_status == "has_new_content":
        print("✅ Deduplication completed, new content found, continue workflow", file=sys.stderr)
        sys.exit(0)
    elif dedup_status == "no_new_content":
        print("⏹️ Deduplication completed, no new content, stop workflow", file=sys.stderr)
        sys.exit(1)
    elif dedup_status == "no_data":
        print("⏹️ No data found, stop workflow", file=sys.stderr)
        sys.exit(1)
    elif dedup_status == "error":
        print("❌ Deduplication processing error, stop workflow", file=sys.stderr)
        sys.exit(2)
    else:
        # Unexpected case: unknown status
        print("❌ Unknown deduplication status, stop workflow", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main() 