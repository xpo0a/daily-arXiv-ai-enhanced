# 灵活arXiv爬虫使用说明

## 功能概述

这个增强版的arXiv爬虫支持根据GitHub Actions中的KEYWORDS变量来爬取指定日期范围的论文，特别适合"今天爬取明天"这样的需求。

## 主要特性

### 🎯 灵活的日期支持
- **相对日期**: `today`, `tomorrow`, `yesterday`, `+1`, `-1`, `+7` 等
- **绝对日期**: `2024-01-01`, `2024/01/01` 等格式
- **日期范围**: 支持指定开始和结束日期

### 🔍 智能关键词搜索
- 支持多个关键词，用逗号分隔
- 自动生成arXiv搜索查询
- 关键词验证和错误处理

### ⚡ 自动化工作流
- 每日定时爬取
- 手动触发支持自定义参数
- 智能去重和AI增强处理

## 配置步骤

### 1. 设置GitHub仓库变量

在仓库的 `Settings -> Secrets and variables -> Actions` 中设置以下变量：

#### Secrets (敏感信息)
- `OPENAI_API_KEY`: 你的OpenAI API密钥
- `OPENAI_BASE_URL`: API基础URL (如: https://api.deepseek.com)

#### Variables (非敏感信息)
- `KEYWORDS`: 关键词，用逗号分隔，如: `machine learning, deep learning, computer vision`
- `LANGUAGE`: 语言，如: `Chinese` 或 `English`
- `MODEL_NAME`: 模型名称，如: `deepseek-chat`
- `EMAIL`: 用于Git提交的邮箱
- `NAME`: 用于Git提交的用户名

#### 可选变量
- `PER_PAGE`: 每页结果数 (默认: 200)
- `MAX_PAGES`: 最大页数 (默认: 10)
- `DATE_FIELD`: 日期字段 (默认: published)

### 2. 工作流文件

使用新的工作流文件: `.github/workflows/arxiv-daily-flexible.yml`

## 使用方法

### 自动运行
工作流会每天UTC时间8:00自动运行，爬取当天的论文。

### 手动触发
1. 进入仓库的 `Actions` 页面
2. 选择 `arXiv Daily Flexible Crawler` 工作流
3. 点击 `Run workflow`
4. 设置参数:
   - **开始日期**: 如 `tomorrow`, `+1`, `2024-01-15`
   - **结束日期**: 如 `tomorrow`, `+2`, `2024-01-16`
   - **强制运行**: 即使没有新内容也运行

## 日期格式说明

### 相对日期
- `today`: 今天
- `tomorrow`: 明天
- `yesterday`: 昨天
- `+1`: 明天 (等同于tomorrow)
- `+2`: 后天
- `-1`: 昨天 (等同于yesterday)
- `-2`: 前天

### 绝对日期
- `2024-01-15`: 标准格式
- `2024/01/15`: 斜杠格式

## 示例用法

### 今天爬取明天的论文
```
开始日期: tomorrow
结束日期: tomorrow
```

### 爬取未来一周的论文
```
开始日期: tomorrow
结束日期: +7
```

### 爬取特定日期范围
```
开始日期: 2024-01-15
结束日期: 2024-01-20
```

### 爬取昨天的论文
```
开始日期: yesterday
结束日期: yesterday
```

## 关键词设置示例

### 机器学习相关
```
KEYWORDS: machine learning, deep learning, neural network, artificial intelligence
```

### 计算机视觉
```
KEYWORDS: computer vision, image recognition, object detection, semantic segmentation
```

### 自然语言处理
```
KEYWORDS: natural language processing, NLP, language model, text generation
```

### 多领域组合
```
KEYWORDS: machine learning, computer vision, NLP, reinforcement learning, generative AI
```

## 输出文件

爬取完成后会生成以下文件:
- `data/YYYY-MM-DD.jsonl`: 原始论文数据
- `data/YYYY-MM-DD_AI_enhanced_Chinese.jsonl`: AI增强后的数据
- `data/YYYY-MM-DD.md`: Markdown格式的摘要

## 故障排除

### 常见问题

1. **关键词验证失败**
   - 检查关键词格式是否正确
   - 确保关键词数量不超过20个
   - 每个关键词长度在2-100字符之间

2. **日期解析失败**
   - 使用支持的日期格式
   - 检查日期范围是否合理 (不超过365天)

3. **爬取结果为空**
   - 检查关键词是否过于具体
   - 确认目标日期是否有论文发布
   - 尝试扩大日期范围

4. **API调用失败**
   - 检查API密钥是否正确
   - 确认API基础URL可访问
   - 检查API配额是否充足

### 调试方法

1. 查看GitHub Actions日志
2. 运行测试脚本: `python test_flexible_crawler.py`
3. 检查环境变量设置

## 技术架构

### 核心组件
- `arxiv_flexible.py`: 灵活爬虫主程序
- `date_validator.py`: 日期验证和解析工具
- `arxiv-daily-flexible.yml`: GitHub Actions工作流

### 数据流程
1. 环境变量验证
2. 日期范围解析
3. 关键词搜索查询生成
4. arXiv API数据爬取
5. 去重检查
6. AI增强处理
7. Markdown转换
8. 自动提交推送

## 更新日志

- **v1.0**: 初始版本，支持基本的关键词和日期爬取
- **v1.1**: 添加日期验证和错误处理
- **v1.2**: 支持相对日期和日期范围
- **v1.3**: 优化搜索查询生成和关键词验证

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

本项目采用Apache-2.0许可证。
