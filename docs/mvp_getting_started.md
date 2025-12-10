# DropQA MVP 快速开始指南

## 功能介绍

DropQA 是一个智能文档问答系统，支持将 Markdown 文档索引后进行自然语言问答。

### 核心功能

1. **文档索引**
   - 自动监控指定目录，发现新文件自动索引
   - 解析 Markdown 层级结构（#/##/### 标题）
   - 构建文档节点树，支持面包屑导航
   - 全文搜索索引（PostgreSQL tsvector）

2. **智能问答**
   - 基于全文搜索检索相关文档片段
   - RAG（检索增强生成）模式调用 LLM
   - 返回回答 + 引用来源（文档名、章节路径）

3. **Web 界面**
   - 简洁的问答界面
   - 实时显示回答和引用来源

### 系统架构

```
┌─────────────────┐     ┌─────────────────┐
│  Indexer 服务    │     │  Server 服务     │
│  (文件监控+索引)  │     │  (API+问答)      │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
           ┌─────────▼─────────┐
           │  Repository 层    │
           │  (存储抽象)        │
           └─────────┬─────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
  ┌──────▼──────┐       ┌───────▼───────┐
  │  PostgreSQL  │       │    SQLite     │
  │  (生产环境)   │       │  (开发/单机)   │
  └─────────────┘       └───────────────┘
```

---

## 系统要求

- Python 3.11+
- **存储后端**（二选一）：
  - PostgreSQL 14+（推荐生产环境）
  - SQLite 3.35+（开发测试、单机部署）
- LLM 服务（OpenAI API 兼容格式）

---

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/hwuu/dropqa.git
cd dropqa
```

### 2. 创建虚拟环境

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

---

## 配置说明

### 1. 创建配置文件

```bash
# 复制示例配置
cp config/indexer.example.yaml config/indexer.yaml
cp config/server.example.yaml config/server.yaml
```

### 2. 配置 Indexer（config/indexer.yaml）

```yaml
# 存储后端配置
storage:
  backend: postgres  # 或 sqlite

  # PostgreSQL 配置（backend: postgres 时使用）
  postgres:
    host: localhost
    port: 5432
    name: dropqa
    user: postgres
    password: your_password

  # SQLite 配置（backend: sqlite 时使用）
  sqlite:
    db_path: ./data/dropqa.db
    chroma_path: ./data/chroma  # 向量搜索预留

# 文件监控配置
watch:
  directories:
    - ~/dropqa_watching    # 监控的目录，支持 ~ 展开
  extensions:
    - .md                  # 监控的文件扩展名

# LLM 配置（Indexer 暂未使用，预留）
llm:
  api_base: http://localhost:11434/v1
  api_key: sk-
  model: Qwen/Qwen3-32B
```

### 3. 配置 Server（config/server.yaml）

```yaml
# HTTP 服务配置
server:
  host: 0.0.0.0
  port: 8000

# 存储后端配置（与 Indexer 保持一致）
storage:
  backend: postgres  # 或 sqlite

  postgres:
    host: localhost
    port: 5432
    name: dropqa
    user: postgres
    password: your_password

  sqlite:
    db_path: ./data/dropqa.db
    chroma_path: ./data/chroma

# LLM 配置
llm:
  api_base: http://localhost:11434/v1   # OpenAI API 兼容地址
  api_key: sk-                          # API Key
  model: Qwen/Qwen3-32B                 # 模型名称
  temperature: 0.2
  max_tokens: 16384
  system_prompt: ""                     # 可选的系统提示

# 检索配置
retrieval:
  top_k: 10                             # 检索文档数量
  relevance_threshold: 0.7
```

### 4. 配置存储后端

#### 方式一：PostgreSQL（推荐生产环境）

确保 PostgreSQL 已安装并运行，创建数据库：

```sql
CREATE DATABASE dropqa;
```

修改配置文件中的 `storage.backend` 为 `postgres`。

#### 方式二：SQLite（开发测试、单机部署）

SQLite 无需安装独立数据库服务，修改配置文件：

```yaml
storage:
  backend: sqlite
  sqlite:
    db_path: ./data/dropqa.db      # 数据库文件路径
    chroma_path: ./data/chroma     # 向量搜索预留
```

SQLite 后端会自动创建数据库文件和目录，无需手动初始化。

**后端对比**：

| 特性 | PostgreSQL | SQLite |
|------|-----------|--------|
| **全文搜索** | tsvector + GIN | FTS5 |
| **中文分词** | pg_jieba / zhparser | unicode61 |
| **部署复杂度** | 需要独立服务 | 单文件 |
| **适用场景** | 生产环境 | 开发测试 |

---

## 启动运行

### 步骤 1：初始化数据库

首次运行需要初始化数据库表和索引：

```bash
python -m dropqa.common.init_db --config config/indexer.yaml
```

输出示例：
```
加载配置: config/indexer.yaml
连接数据库: localhost:5432/dropqa
创建数据库表...
  ✓ 表创建完成
创建全文搜索索引...
  ✓ 全文搜索索引创建完成

数据库初始化完成！
```

### 步骤 2：创建监控目录

```bash
# 创建默认监控目录
mkdir ~/dropqa_watching
```

### 步骤 3：启动 Indexer 服务

```bash
python -m dropqa.indexer --config config/indexer.yaml
```

输出示例：
```
2024-12-08 10:00:00 [INFO] 加载配置: config/indexer.yaml
2024-12-08 10:00:00 [INFO] 连接数据库: localhost:5432/dropqa
2024-12-08 10:00:00 [INFO] 开始扫描现有文件...
2024-12-08 10:00:00 [INFO] 扫描完成，共索引 0 个文件
2024-12-08 10:00:00 [INFO] 开始监控: C:\Users\xxx\dropqa_watching
2024-12-08 10:00:00 [INFO] Indexer 服务已启动，按 Ctrl+C 退出
```

### 步骤 4：启动 Server 服务

打开新终端窗口：

```bash
python -m dropqa.server --config config/server.yaml
```

输出示例：
```
2024-12-08 10:00:00 [INFO] 加载配置: config/server.yaml
2024-12-08 10:00:00 [INFO] 启动服务: 0.0.0.0:8000
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 步骤 5：添加测试文档

将 Markdown 文件复制到监控目录：

```bash
cp your_document.md ~/dropqa_watching/
```

Indexer 会自动检测并索引文件。

---

## 使用方法

### Web 界面

打开浏览器访问：http://localhost:8000/

1. 在输入框输入问题
2. 点击"提问"或按回车
3. 查看回答和引用来源

### API 接口

**问答接口**

```bash
curl -X POST http://localhost:8000/api/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "项目背景是什么？"}'
```

**响应示例**

```json
{
  "answer": "根据文档，项目背景是...",
  "sources": [
    {
      "document_name": "report.md",
      "path": "第1章 > 1.1 背景",
      "content_snippet": "这是相关内容片段..."
    }
  ]
}
```

**健康检查**

```bash
curl http://localhost:8000/health
# 返回: {"status": "ok"}
```

---

## 常见问题

### 1. 数据库连接失败

检查：
- PostgreSQL 是否启动
- 数据库名、用户名、密码是否正确
- 防火墙是否允许连接

### 2. LLM 服务连接失败

检查：
- LLM 服务是否启动
- `api_base` 地址是否正确
- `api_key` 是否有效

### 3. 文件没有被索引

检查：
- 文件是否在监控目录内
- 文件扩展名是否在配置的 `extensions` 列表中
- 文件名是否以 `.` 开头（隐藏文件会被跳过）

### 4. 搜索没有结果

检查：
- 文档是否已被索引（查看 Indexer 日志）
- 搜索关键词是否在文档内容中

---

## 目录结构

```
dropqa/
├── config/
│   ├── indexer.example.yaml    # Indexer 配置示例
│   └── server.example.yaml     # Server 配置示例
├── dropqa/
│   ├── common/                 # 公共模块
│   │   ├── config.py           # 配置管理
│   │   ├── db.py               # 数据库连接
│   │   ├── models.py           # 数据模型
│   │   ├── init_db.py          # 数据库初始化
│   │   └── repository/         # Repository 抽象层
│   │       ├── __init__.py
│   │       ├── base.py         # 接口定义
│   │       ├── postgres.py     # PostgreSQL 实现
│   │       └── sqlite.py       # SQLite 实现
│   ├── indexer/                # Indexer 服务
│   │   ├── __main__.py         # 入口
│   │   ├── parser.py           # Markdown 解析
│   │   ├── indexer.py          # 索引写入
│   │   └── watcher.py          # 文件监控
│   └── server/                 # Server 服务
│       ├── __main__.py         # 入口
│       ├── app.py              # FastAPI 应用
│       ├── search.py           # 全文搜索
│       ├── llm.py              # LLM 服务
│       ├── qa.py               # RAG 问答
│       └── static/
│           └── index.html      # 前端页面
├── tests/                      # 单元测试
├── docs/                       # 文档
├── requirements.txt            # 依赖
└── pyproject.toml              # 项目配置
```

---

## 下一步

MVP 验证通过后，可以考虑：

1. **支持更多文档格式**：Word、PDF、Excel
2. **向量搜索**：集成 pgvector 进行语义搜索
3. **多轮对话**：支持上下文连续问答
4. **用户管理**：多用户、权限控制
5. **文档管理界面**：上传、删除、查看索引状态
