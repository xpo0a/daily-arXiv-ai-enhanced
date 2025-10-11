#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试日期计算逻辑
"""
import sys
import os
from datetime import datetime, timedelta

sys.path.append('daily_arxiv')

from daily_arxiv.utils.date_validator import DateValidator

def test_date_calculation():
    """测试日期计算逻辑"""
    
    print("Testing date calculation logic")
    print("=" * 50)
    
    # 当前时间（模拟GitHub Actions运行时间）
    now = datetime(2025, 10, 11, 8, 19, 48)  # 2025-10-11 08:19:48
    print(f"Current time: {now}")
    print(f"Current date: {now.date()}")
    print(f"Current weekday: {now.weekday()} (0=Monday, 6=Sunday)")
    print()
    
    # 手动计算上一周
    today = now.date()
    days_since_monday = today.weekday()  # 0=Monday, 6=Sunday
    print(f"Days since Monday: {days_since_monday}")
    
    # 计算本周一和本周日
    this_monday = today - timedelta(days=days_since_monday)
    this_sunday = this_monday + timedelta(days=6)
    print(f"This week Monday: {this_monday}")
    print(f"This week Sunday: {this_sunday}")
    
    # 计算上一周
    last_monday = this_monday - timedelta(days=7)
    last_sunday = this_sunday - timedelta(days=7)
    print(f"Last week Monday: {last_monday}")
    print(f"Last week Sunday: {last_sunday}")
    print()
    
    # 测试DateValidator
    print("Testing DateValidator:")
    print("-" * 30)
    
    # 测试last_monday
    last_monday_result = DateValidator.parse_date_input('last_monday')
    print(f"last_monday result: {last_monday_result.date()}")
    
    # 测试last_sunday
    last_sunday_result = DateValidator.parse_date_input('last_sunday')
    print(f"last_sunday result: {last_sunday_result.date()}")
    print()
    
    # 验证结果
    print("Verification:")
    print("-" * 30)
    expected_monday = last_monday
    expected_sunday = last_sunday
    
    actual_monday = last_monday_result.date()
    actual_sunday = last_sunday_result.date()
    
    print(f"Expected Monday: {expected_monday}")
    print(f"Actual Monday:   {actual_monday}")
    print(f"Match: {expected_monday == actual_monday}")
    print()
    
    print(f"Expected Sunday: {expected_sunday}")
    print(f"Actual Sunday:   {actual_sunday}")
    print(f"Match: {expected_sunday == actual_sunday}")
    print()
    
    # 分析问题
    print("Problem Analysis:")
    print("-" * 30)
    print(f"From logs: target_range=2025-09-29 to 2025-10-05")
    print(f"Expected range: {last_monday} to {last_sunday}")
    print()
    
    if actual_monday != expected_monday or actual_sunday != expected_sunday:
        print("❌ Date calculation is incorrect!")
        print("The issue is in the DateValidator.parse_date_input method")
    else:
        print("✅ Date calculation is correct!")
        print("The issue might be elsewhere in the code")

if __name__ == "__main__":
    test_date_calculation()
