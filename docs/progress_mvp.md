# DropQA MVP 开发进度

## 目标

快速验证核心价值：能上传 Markdown 文档、能问答、能看到引用来源。

## 验收标准

- [ ] 能上传 Markdown 文件，自动解析层级结构
- [ ] 能用自然语言提问，获得基于文档的回答
- [ ] 回答中包含引用的文档名和章节路径
- [ ] 端到端流程跑通

## 开发计划

### 1. 基础设施

| 任务 | 状态 | 说明 |
|------|------|------|
| 1.1 项目目录结构 | [x] | dropqa/indexer, dropqa/server, dropqa/common |
| 1.2 依赖管理 | [x] | requirements.txt, pyproject.toml |
| 1.3 配置管理 | [x] | config.py, indexer.example.yaml, server.example.yaml |
| 1.4 数据库连接 | [x] | db.py, SQLAlchemy 异步连接 |
| 1.5 数据模型 | [x] | models.py (documents, nodes 表) |
| 1.6 数据库初始化脚本 | [x] | init_db.py, 建表 + tsvector 触发器 + GIN 索引 |

### 2. Indexer 服务

| 任务 | 状态 | 说明 |
|------|------|------|
| 2.1 文件监控 | [x] | watcher.py, watchdog 监控目录 |
| 2.2 Markdown 解析 | [x] | parser.py, 解析标题层级 (#/##/###) |
| 2.3 节点树构建 | [x] | parser.py flatten_nodes(), 构建 parent-child 关系 |
| 2.4 索引写入 | [x] | indexer.py, 写入 documents, nodes 表 |
| 2.5 全文索引 | [x] | init_db.py, tsvector 触发器自动更新 |
| 2.6 服务入口 | [x] | __main__.py, CLI 启动 |

### 3. Server 服务

| 任务 | 状态 | 说明 |
|------|------|------|
| 3.1 FastAPI 框架 | [x] | app.py, __main__.py |
| 3.2 全文搜索 | [x] | search.py, tsvector 查询 |
| 3.3 Breadcrumb 构建 | [x] | search.py get_node_context() |
| 3.4 LLM 服务 | [x] | llm.py, OpenAI API 封装 |
| 3.5 RAG 问答 | [x] | qa.py, Prompt 模板 + 引用格式化 |
| 3.6 API 接口 | [x] | POST /api/qa/ask |
| 3.7 简单前端 | [x] | static/index.html |

---

## 当前进度

### 2024-12-08

**MVP 开发完成！**

已完成：
- 项目目录结构（dropqa/indexer, dropqa/server, dropqa/common）
- 依赖管理（requirements.txt, pyproject.toml）
- 配置管理（config.py + 8 个单元测试）
- 数据库连接模块（db.py）
- 数据模型（models.py: Document, Node）
- Pydantic 模型（schemas.py）
- 配置示例文件（indexer.example.yaml, server.example.yaml）
- 数据库初始化脚本（init_db.py: 建表 + tsvector 触发器 + GIN 索引）
- Markdown 解析器（parser.py + 9 个单元测试）
- 节点树展平（flatten_nodes 函数）
- 索引写入（indexer.py + 7 个单元测试）
- 文件监控（watcher.py + 11 个单元测试）
- Indexer 服务入口（__main__.py）
- FastAPI 框架（app.py + 7 个单元测试）
- 全文搜索（search.py + 13 个单元测试）
- LLM 服务（llm.py + 8 个单元测试）
- RAG 问答（qa.py + 8 个单元测试）
- API 接口（POST /api/qa/ask）
- 简单前端（static/index.html）

单元测试：71 passed

待验收：
- 端到端测试（需要真实数据库和 LLM 服务）

---

## 技术决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| 异步框架 | asyncio + asyncpg | FastAPI 原生支持，性能好 |
| ORM | SQLAlchemy 2.0 async | 成熟、类型支持好 |
| 配置格式 | YAML | 可读性好，支持环境变量 |
| 中文分词 | pg_jieba | PostgreSQL 原生支持，性能好 |

---

## 已知问题

暂无
