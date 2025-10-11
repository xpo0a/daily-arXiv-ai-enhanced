# 快速开始指南

## 🚀 立即使用灵活arXiv爬虫

### 1. 设置GitHub仓库变量

在仓库的 `Settings -> Secrets and variables -> Actions` 中设置：

#### Secrets (敏感信息)
```
OPENAI_API_KEY: 你的API密钥
OPENAI_BASE_URL: https://api.deepseek.com
```

#### Variables (非敏感信息)
```
KEYWORDS: machine learning, deep learning, computer vision
LANGUAGE: Chinese
MODEL_NAME: deepseek-chat
EMAIL: your-email@example.com
NAME: Your Name
```

### 2. 手动触发爬取

1. 进入仓库的 `Actions` 页面
2. 选择 `arXiv Daily Flexible Crawler` 工作流
3. 点击 `Run workflow`
4. 设置参数：
   - **开始日期**: `tomorrow` (爬取明天的论文)
   - **结束日期**: `tomorrow`
   - **强制运行**: `false`

### 3. 常用日期格式

| 格式 | 说明 | 示例 |
|------|------|------|
| `today` | 今天 | 爬取今天的论文 |
| `tomorrow` | 明天 | 爬取明天的论文 |
| `yesterday` | 昨天 | 爬取昨天的论文 |
| `+1` | 明天 | 等同于tomorrow |
| `+7` | 一周后 | 爬取一周后的论文 |
| `2024-01-15` | 绝对日期 | 爬取指定日期的论文 |

### 4. 关键词设置示例

#### 机器学习
```
KEYWORDS: machine learning, deep learning, neural network, artificial intelligence
```

#### 计算机视觉
```
KEYWORDS: computer vision, image recognition, object detection, semantic segmentation
```

#### 自然语言处理
```
KEYWORDS: natural language processing, NLP, language model, text generation
```

#### 多领域
```
KEYWORDS: machine learning, computer vision, NLP, reinforcement learning
```

### 5. 查看结果

爬取完成后，在 `data/` 目录下会生成：
- `YYYY-MM-DD.jsonl`: 原始论文数据
- `YYYY-MM-DD_AI_enhanced_Chinese.jsonl`: AI增强数据
- `YYYY-MM-DD.md`: Markdown摘要

### 6. 自动运行

工作流会每天UTC时间8:00自动运行，爬取当天的论文。

## 🔧 故障排除

### 常见问题

1. **爬取结果为空**
   - 检查关键词是否过于具体
   - 尝试扩大日期范围
   - 确认目标日期有论文发布

2. **API调用失败**
   - 检查API密钥是否正确
   - 确认API基础URL可访问
   - 检查API配额

3. **日期解析失败**
   - 使用支持的日期格式
   - 检查日期范围是否合理

### 获取帮助

- 查看详细文档: `FLEXIBLE_CRAWLER_README.md`
- 运行测试: `python test_flexible_crawler.py`
- 提交Issue获取支持

## 📝 示例工作流

### 今天爬取明天的论文
```
开始日期: tomorrow
结束日期: tomorrow
关键词: machine learning, deep learning
```

### 爬取未来一周的论文
```
开始日期: tomorrow
结束日期: +7
关键词: computer vision, image processing
```

### 爬取特定日期范围
```
开始日期: 2024-01-15
结束日期: 2024-01-20
关键词: natural language processing, NLP
```

---

**开始使用吧！** 🎉
