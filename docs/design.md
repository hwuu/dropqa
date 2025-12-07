# DropQA 文档问答系统设计文档

## 目录

- [1. 背景](#1-背景)
- [2. 需求概述](#2-需求概述)
- [3. 架构设计](#3-架构设计)
  - [3.1 整体架构](#31-整体架构)
  - [3.2 技术栈](#32-技术栈)
  - [3.3 模块划分](#33-模块划分)
- [4. 数据结构设计](#4-数据结构设计)
  - [4.1 数据库表设计](#41-数据库表设计)
  - [4.2 向量索引设计](#42-向量索引设计)
  - [4.3 层级节点结构](#43-层级节点结构)
- [5. 核心算法设计](#5-核心算法设计)
  - [5.1 文档解析流程](#51-文档解析流程)
  - [5.2 层级索引构建](#52-层级索引构建)
  - [5.3 话题自动提取与聚类](#53-话题自动提取与聚类)
  - [5.4 RAG 检索与问答](#54-rag-检索与问答)
  - [5.5 版本管理](#55-版本管理)
- [6. 接口设计](#6-接口设计)
  - [6.1 后端 API](#61-后端-api)
  - [6.2 前端页面](#62-前端页面)
- [7. 测试方案](#7-测试方案)
- [8. 开发计划](#8-开发计划)
- [9. 扩展方向](#9-扩展方向)
- [10. 附录](#10-附录)

---

## 1. 背景

用户拥有大量不同格式的文档（PPT、Word、Excel、Markdown 等），需要一个智能问答系统来：
- 快速检索定位相关文档
- 针对特定话题进行问答
- 对文档或话题自动生成摘要

文档会持续更新、添加和删除，需要系统能够自动管理索引，无需手动维护目录结构。

---

## 2. 需求概述

| 类别 | 需求描述 |
|------|----------|
| **文档管理** | 支持 PPT、Word、Excel、Markdown 等格式 |
| | Web 拖拽上传，自动解析入库 |
| | 增量更新索引，支持文档增删改 |
| | 版本追踪（简单版：保留历史版本，可按版本问答） |
| **智能问答** | 基于 RAG 的文档问答 |
| | 回答时引用来源（文档名 + 章节位置） |
| | 实时生成摘要 |
| **话题管理** | 自动提取文档话题标签 |
| | 基于话题自动聚类（一个文档可属于多个话题） |
| | 问答时自动识别问题所属话题领域 |
| **多模态** | 图片 OCR（PaddleOCR） + 多模态理解 |
| | 入库时全量处理图片 |
| **部署** | 个人使用，需支持远程访问 |
| | 文档量级：上千个 |

---

## 3. 架构设计

### 3.1 整体架构

```
+-----------------------------------------------------------+
|                      Web Frontend                          |
|                  (Vue 3 + TailwindCSS)                     |
|  +---------------+  +---------------+  +----------------+  |
|  | Upload/Manage |  | Topic Browser |  | Q&A Interface  |  |
|  +---------------+  +---------------+  +----------------+  |
+-----------------------------------------------------------+
                             |
                             | HTTP/REST
                             v
+-----------------------------------------------------------+
|                     Backend API                            |
|                      (FastAPI)                             |
|  +---------------+  +---------------+  +----------------+  |
|  | Doc Manager   |  | Topic Service |  | QA Service     |  |
|  +---------------+  +---------------+  +----------------+  |
+-----------------------------------------------------------+
          |                    |                    |
          v                    v                    v
+-------------------+  +----------------+  +----------------+
| Document Pipeline |  | Index Engine   |  | LLM Service    |
| - Parser          |  | (LlamaIndex)   |  | (OpenAI API)   |
| - OCR             |  |                |  |                |
| - Chunker         |  |                |  |                |
+-------------------+  +----------------+  +----------------+
          |                    |
          v                    v
+-----------------------------------------------------------+
|                    PostgreSQL + pgvector                   |
|  +---------------+  +---------------+  +----------------+  |
|  | documents     |  | nodes         |  | topics         |  |
|  | versions      |  | embeddings    |  | doc_topics     |  |
+-----------------------------------------------------------+
```

### 3.2 技术栈

| 层级 | 技术选型 | 说明 |
|------|----------|------|
| **前端** | Vue 3 + TailwindCSS | 现代美观，组件化开发 |
| **后端** | FastAPI | 高性能，异步支持好 |
| **索引框架** | LlamaIndex | 层级索引、文档管理支持好 |
| **向量数据库** | PostgreSQL + pgvector | 成熟稳定，元数据与向量统一存储 |
| **文档解析** | unstructured | 统一多格式解析接口 |
| **OCR** | PaddleOCR | 中文 OCR 效果好 |
| **多模态** | OpenAI Vision API 或兼容接口 | 图片理解 |
| **嵌入模型** | OpenAI API 格式 | 可接本地或云端 |
| **LLM** | OpenAI API 格式 | 可接本地或云端 |

### 3.3 模块划分

```
dropqa/
├── backend/
│   ├── api/                  # FastAPI 路由
│   │   ├── documents.py      # 文档上传、管理接口
│   │   ├── topics.py         # 话题相关接口
│   │   └── qa.py             # 问答接口
│   ├── services/
│   │   ├── document_service.py   # 文档管理业务逻辑
│   │   ├── parser_service.py     # 文档解析服务
│   │   ├── index_service.py      # 索引构建服务
│   │   ├── topic_service.py      # 话题提取与聚类
│   │   ├── qa_service.py         # RAG 问答服务
│   │   └── llm_service.py        # LLM/Embedding 调用封装
│   ├── models/
│   │   ├── database.py       # 数据库连接
│   │   └── schemas.py        # Pydantic 模型
│   ├── core/
│   │   ├── config.py         # 配置管理
│   │   └── ocr.py            # PaddleOCR 封装
│   └── main.py               # FastAPI 入口
├── frontend/
│   ├── src/
│   │   ├── views/            # 页面组件
│   │   ├── components/       # 通用组件
│   │   ├── api/              # API 调用封装
│   │   └── stores/           # Pinia 状态管理
│   └── ...
├── tests/
│   └── unit/                 # 单元测试
├── docs/
│   └── design.md             # 本文档
└── requirements.txt
```

---

## 4. 数据结构设计

### 4.1 数据库表设计

#### documents 表（文档主表）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| filename | VARCHAR(500) | 原始文件名 |
| file_type | VARCHAR(50) | 文件类型（pptx/docx/xlsx/md） |
| file_hash | VARCHAR(64) | 文件 SHA256 哈希 |
| file_size | BIGINT | 文件大小（字节） |
| storage_path | VARCHAR(1000) | 存储路径 |
| current_version | INTEGER | 当前版本号 |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

#### document_versions 表（版本表）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| document_id | UUID | 关联文档 |
| version | INTEGER | 版本号 |
| file_hash | VARCHAR(64) | 该版本文件哈希 |
| storage_path | VARCHAR(1000) | 该版本文件存储路径 |
| created_at | TIMESTAMP | 版本创建时间 |

#### nodes 表（层级节点表）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| document_id | UUID | 关联文档 |
| version | INTEGER | 关联版本 |
| parent_id | UUID | 父节点（NULL 表示根节点） |
| node_type | VARCHAR(50) | 节点类型（document/section/paragraph） |
| title | VARCHAR(500) | 标题（章节名） |
| content | TEXT | 文本内容 |
| position | INTEGER | 同级排序 |
| metadata | JSONB | 扩展元数据（页码、图片描述等） |
| created_at | TIMESTAMP | 创建时间 |

#### embeddings 表（向量表）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| node_id | UUID | 关联节点 |
| embedding | VECTOR(1536) | 向量（维度取决于模型） |
| model_name | VARCHAR(100) | 使用的嵌入模型 |
| created_at | TIMESTAMP | 创建时间 |

#### topics 表（话题表）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| name | VARCHAR(200) | 话题名称 |
| description | TEXT | 话题描述 |
| embedding | VECTOR(1536) | 话题名称的嵌入（用于相似合并） |
| created_at | TIMESTAMP | 创建时间 |

#### document_topics 表（文档-话题关联表）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| document_id | UUID | 关联文档 |
| topic_id | UUID | 关联话题 |
| confidence | FLOAT | 相关度分数 |
| created_at | TIMESTAMP | 创建时间 |

#### images 表（图片表）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| node_id | UUID | 关联节点 |
| storage_path | VARCHAR(1000) | 图片存储路径 |
| ocr_text | TEXT | OCR 提取文本 |
| description | TEXT | 多模态理解描述 |
| embedding | VECTOR(1536) | 图片描述的嵌入 |
| created_at | TIMESTAMP | 创建时间 |

### 4.2 向量索引设计

使用 pgvector 扩展，在 embeddings 表上建立 IVFFlat 或 HNSW 索引：

```sql
-- 启用 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- HNSW 索引（推荐，查询更快）
CREATE INDEX ON embeddings USING hnsw (embedding vector_cosine_ops);

-- 或 IVFFlat 索引（构建更快，适合频繁更新）
CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### 4.3 层级节点结构

文档解析后形成树状结构：

```
Document (根节点)
├── Section: "第1章 概述"
│   ├── Paragraph: "1.1 背景介绍..."
│   │   └── Image: [图片OCR文本 + 多模态描述]
│   └── Paragraph: "1.2 目标..."
├── Section: "第2章 详细设计"
│   └── ...
```

检索策略（LlamaIndex HierarchicalNodeParser）：
- **检索单位**：Paragraph（段落级）
- **上下文增强**：检索到段落时，自动带上其所属 Section 的标题作为上下文
- **引用定位**：`文档名 > 章节名 > 段落位置`

---

## 5. 核心算法设计

### 5.1 文档解析流程

```
+-------------+     +---------------+     +----------------+
| Upload File | --> | Detect Type   | --> | Parse Content  |
+-------------+     +---------------+     +----------------+
                                                  |
                    +-----------------------------+
                    |
          +---------+---------+
          |                   |
          v                   v
+------------------+  +------------------+
| Extract Text     |  | Extract Images   |
| (unstructured)   |  | (unstructured)   |
+------------------+  +------------------+
          |                   |
          v                   v
+------------------+  +------------------+
| Hierarchical     |  | OCR + Multimodal |
| Chunking         |  | (PaddleOCR+LLM)  |
+------------------+  +------------------+
          |                   |
          +--------+----------+
                   |
                   v
          +------------------+
          | Generate         |
          | Embeddings       |
          +------------------+
                   |
                   v
          +------------------+
          | Store to DB      |
          +------------------+
```

各格式解析要点：

| 格式 | 层级识别方式 | 图片处理 |
|------|--------------|----------|
| DOCX | 标题样式（Heading 1/2/3） | 内嵌图片提取 |
| PPTX | 每页为一个 Section | 幻灯片图片/图表 |
| XLSX | 每个 Sheet 为一个 Section | 图表转图片处理 |
| Markdown | 标题层级（#/##/###） | 引用图片路径 |

### 5.2 层级索引构建

使用 LlamaIndex 的 `HierarchicalNodeParser`：

```python
from llama_index.core.node_parser import HierarchicalNodeParser

# Configure parser
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128],  # document, section, paragraph
    chunk_overlap=20
)

# Parse document into hierarchical nodes
nodes = node_parser.get_nodes_from_documents(documents)
```

索引构建伪代码：

```
function build_index(document):
    # 1. Parse document structure
    sections = parse_document_structure(document)

    # 2. Build node tree
    root_node = create_node(type="document", title=document.name)

    for section in sections:
        section_node = create_node(
            type="section",
            title=section.title,
            parent=root_node
        )

        for paragraph in section.paragraphs:
            para_node = create_node(
                type="paragraph",
                content=paragraph.text,
                parent=section_node
            )

            # Generate embedding for paragraph
            embedding = embed(paragraph.text)
            store_embedding(para_node, embedding)

            # Process images in paragraph
            for image in paragraph.images:
                process_image(image, para_node)

    # 3. Store to database
    save_nodes_to_db(root_node)
```

### 5.3 话题自动提取与聚类

#### 5.3.1 文档入库时提取话题

```
function extract_topics(document):
    # Use LLM to extract 3-5 topic tags
    prompt = """
    Based on the document content, extract 3-5 topic tags.
    Return as JSON: {"topics": ["topic1", "topic2", ...]}

    Document content:
    {document_summary}
    """

    topics = llm.generate(prompt)

    for topic in topics:
        # Check if similar topic exists
        topic_embedding = embed(topic)
        similar = find_similar_topics(topic_embedding, threshold=0.85)

        if similar:
            # Link to existing topic
            link_document_to_topic(document, similar[0])
        else:
            # Create new topic
            new_topic = create_topic(topic, topic_embedding)
            link_document_to_topic(document, new_topic)
```

#### 5.3.2 问答时识别话题

```
function identify_question_topic(question):
    question_embedding = embed(question)

    # Find top-k similar topics
    similar_topics = vector_search(
        table="topics",
        query=question_embedding,
        top_k=3
    )

    return similar_topics
```

### 5.4 RAG 检索与问答

```
+----------+     +----------------+     +------------------+
| Question | --> | Topic Identify | --> | Filter Documents |
+----------+     +----------------+     +------------------+
                                                |
                                                v
                                       +------------------+
                                       | Vector Search    |
                                       | (paragraphs)     |
                                       +------------------+
                                                |
                                                v
                                       +------------------+
                                       | Retrieve Parent  |
                                       | Context (section)|
                                       +------------------+
                                                |
                                                v
                                       +------------------+
                                       | LLM Generate     |
                                       | with Citations   |
                                       +------------------+
                                                |
                                                v
                                       +------------------+
                                       | Answer + Sources |
                                       +------------------+
```

检索问答伪代码：

```
function answer_question(question, version=None):
    # 1. Embed question
    query_embedding = embed(question)

    # 2. Identify relevant topics (optional filter)
    topics = identify_question_topic(question)

    # 3. Vector search for relevant paragraphs
    version_filter = version or "latest"

    results = vector_search(
        table="embeddings",
        query=query_embedding,
        top_k=10,
        filter={
            "version": version_filter,
            "topics": topics  # optional
        }
    )

    # 4. Retrieve parent context for each result
    contexts = []
    for result in results:
        node = get_node(result.node_id)
        parent_section = get_parent_section(node)

        contexts.append({
            "content": node.content,
            "section_title": parent_section.title,
            "document_name": get_document_name(node),
            "position": node.position
        })

    # 5. Generate answer with LLM
    prompt = build_qa_prompt(question, contexts)
    answer = llm.generate(prompt)

    # 6. Format citations
    return {
        "answer": answer,
        "sources": format_citations(contexts)
    }
```

引用格式示例：

```json
{
  "answer": "根据文档记载，项目的主要目标是...",
  "sources": [
    {
      "document": "项目规划.docx",
      "section": "第1章 概述 > 1.2 项目目标",
      "relevance": 0.92
    },
    {
      "document": "需求文档.pptx",
      "section": "第3页 核心需求",
      "relevance": 0.87
    }
  ]
}
```

### 5.5 版本管理

#### 文档更新流程

```
function update_document(document_id, new_file):
    old_doc = get_document(document_id)
    new_hash = calculate_hash(new_file)

    # Check if file actually changed
    if new_hash == old_doc.file_hash:
        return "No changes detected"

    # Create new version
    new_version = old_doc.current_version + 1

    # Store new version file
    storage_path = store_file(new_file, document_id, new_version)

    # Create version record
    create_version_record(
        document_id=document_id,
        version=new_version,
        file_hash=new_hash,
        storage_path=storage_path
    )

    # Parse and index new version (nodes marked with version)
    parse_and_index(new_file, document_id, new_version)

    # Update document current version
    update_document_version(document_id, new_version)

    # Re-extract topics (may have changed)
    extract_topics(document_id, new_version)
```

#### 版本查询

```sql
-- 查询特定版本的节点
SELECT n.*, e.embedding
FROM nodes n
JOIN embeddings e ON n.id = e.node_id
WHERE n.document_id = :doc_id
  AND n.version = :version;

-- 查询最新版本
SELECT n.*, e.embedding
FROM nodes n
JOIN embeddings e ON n.id = e.node_id
JOIN documents d ON n.document_id = d.id
WHERE n.document_id = :doc_id
  AND n.version = d.current_version;
```

---

## 6. 接口设计

### 6.1 后端 API

#### 文档管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/documents/upload | 上传文档（支持批量） |
| GET | /api/documents | 获取文档列表 |
| GET | /api/documents/{id} | 获取文档详情 |
| DELETE | /api/documents/{id} | 删除文档 |
| GET | /api/documents/{id}/versions | 获取文档版本列表 |
| POST | /api/documents/{id}/reindex | 重新索引文档 |

#### 话题管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/topics | 获取话题列表 |
| GET | /api/topics/{id} | 获取话题详情及关联文档 |
| GET | /api/topics/{id}/documents | 获取话题下的文档 |
| POST | /api/topics/merge | 合并相似话题 |

#### 问答

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/qa/ask | 提问 |
| POST | /api/qa/summarize | 生成摘要 |
| GET | /api/qa/history | 获取问答历史 |

#### 请求/响应示例

**POST /api/documents/upload**

```
Request: multipart/form-data
- files: File[] (多个文件)

Response:
{
  "success": true,
  "documents": [
    {
      "id": "uuid",
      "filename": "项目规划.docx",
      "status": "processing"
    }
  ]
}
```

**POST /api/qa/ask**

```json
Request:
{
  "question": "项目的核心目标是什么？",
  "topic_filter": ["uuid"],  // 可选，限定话题
  "version": null            // 可选，null 表示最新版本
}

Response:
{
  "answer": "根据文档记载，项目的核心目标包括三个方面...",
  "sources": [
    {
      "document_id": "uuid",
      "document_name": "项目规划.docx",
      "section": "第1章 概述 > 1.2 项目目标",
      "snippet": "...核心目标是建立一个...",
      "relevance": 0.92
    }
  ],
  "identified_topics": ["项目管理", "规划"]
}
```

### 6.2 前端页面

| 页面 | 功能 |
|------|------|
| **首页/仪表盘** | 文档统计、最近上传、热门话题 |
| **文档管理** | 拖拽上传、文档列表、搜索过滤、版本查看 |
| **话题浏览** | 话题卡片/标签云、点击查看关联文档 |
| **问答界面** | 对话式问答、话题/版本筛选、引用来源展示 |
| **文档详情** | 文档结构树、版本历史、关联话题 |

---

## 7. 测试方案

### 7.1 单元测试

| 模块 | 测试要点 |
|------|----------|
| parser_service | 各格式解析正确性、层级结构提取 |
| index_service | 节点创建、嵌入生成、存储正确性 |
| topic_service | 话题提取、相似度匹配、聚类逻辑 |
| qa_service | 检索准确性、上下文拼接、引用格式 |
| version | 版本创建、版本过滤查询 |

### 7.2 集成测试

| 场景 | 测试要点 |
|------|----------|
| 文档上传流程 | 上传 → 解析 → 索引 → 话题提取 完整流程 |
| 问答流程 | 提问 → 检索 → 生成 → 引用 完整流程 |
| 版本更新 | 上传新版本 → 增量索引 → 版本查询 |

### 7.3 测试数据

准备测试文档集：
- 各格式各 2-3 个文档
- 包含中英文内容
- 包含图片、表格
- 包含明确的层级结构

---

## 8. 开发计划

按依赖关系排序的开发步骤：

### 第一阶段：基础设施

1. **项目初始化**
   - 后端 FastAPI 项目结构
   - 前端 Vue 3 项目结构
   - PostgreSQL + pgvector 环境搭建

2. **数据库层**
   - 表结构创建
   - 数据库连接封装
   - 基础 CRUD 操作

### 第二阶段：文档处理

3. **文档解析服务**
   - unstructured 集成
   - 各格式解析实现
   - 层级结构提取

4. **图片处理**
   - PaddleOCR 集成
   - 多模态理解接口
   - 图片描述生成

5. **索引服务**
   - LlamaIndex 集成
   - 层级节点构建
   - 嵌入生成与存储

### 第三阶段：问答功能

6. **RAG 检索**
   - 向量检索实现
   - 上下文增强
   - 版本过滤

7. **LLM 问答**
   - LLM 服务封装
   - Prompt 模板
   - 引用格式化

### 第四阶段：话题功能

8. **话题提取**
   - LLM 提取话题
   - 话题相似度合并

9. **话题聚类**
   - 问题话题识别
   - 话题筛选功能

### 第五阶段：前端开发

10. **基础页面**
    - 布局框架
    - 文档上传组件
    - 文档列表

11. **问答界面**
    - 对话组件
    - 引用展示
    - 话题/版本筛选

12. **话题浏览**
    - 话题列表
    - 关联文档展示

### 第六阶段：完善

13. **版本管理 UI**
14. **性能优化**
15. **部署配置**

---

## 9. 扩展方向

| 方向 | 说明 |
|------|------|
| **复杂版本管理** | diff 对比、跨版本问答（"这个文档改了什么"） |
| **协作功能** | 多用户、权限管理 |
| **更多格式** | PDF、网页、邮件等 |
| **知识图谱** | 实体抽取、关系构建 |
| **主动推荐** | 基于用户行为推荐相关文档 |
| **定时任务** | 监控文件夹自动同步 |

---

## 10. 附录

### 10.1 pgvector 安装

```bash
# PostgreSQL 需要 pgvector 扩展
# Docker 方式
docker run -d \
  --name postgres-vector \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 10.2 依赖列表（初步）

```
# Backend
fastapi
uvicorn
sqlalchemy
asyncpg
pgvector
llama-index
unstructured[all-docs]
paddleocr
paddlepaddle
openai
python-multipart
pydantic

# Frontend
vue@3
tailwindcss
axios
@vueuse/core
```

### 10.3 参考资料

- [LlamaIndex 文档](https://docs.llamaindex.ai/)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [unstructured 文档](https://unstructured-io.github.io/unstructured/)
- [PaddleOCR 文档](https://paddlepaddle.github.io/PaddleOCR/)
