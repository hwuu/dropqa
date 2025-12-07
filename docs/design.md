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
  - [5.4 版本管理](#54-版本管理)
- [6. Agentic RAG 系统](#6-agentic-rag-系统)
  - [6.1 设计理念](#61-设计理念)
  - [6.2 工具体系](#62-工具体系)
  - [6.3 反馈与自适应](#63-反馈与自适应)
  - [6.4 执行框架](#64-执行框架)
  - [6.5 标准 RAG 流程](#65-标准-rag-流程)
  - [6.6 响应格式](#66-响应格式)
- [7. 接口设计](#7-接口设计)
  - [7.1 后端 API](#71-后端-api)
  - [7.2 前端页面](#72-前端页面)
- [8. 测试方案](#8-测试方案)
  - [8.1 单元测试](#81-单元测试)
  - [8.2 集成测试](#82-集成测试)
  - [8.3 测试数据](#83-测试数据)
- [9. 开发计划](#9-开发计划)
- [10. 扩展方向](#10-扩展方向)
- [11. 附录](#11-附录)
  - [11.1 配置项](#111-配置项)
  - [11.2 pgvector 安装](#112-pgvector-安装)
  - [11.3 依赖列表](#113-依赖列表)
  - [11.4 参考资料](#114-参考资料)

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
| **智能问答** | 基于 Agentic RAG 的文档问答 |
| | 简单问题快速回答，复杂问题自动启动深度推理 |
| | 支持多跳推理（跨文档关联信息） |
| | 回答时引用来源（文档名 + 章节路径） |
| | 推理过程可追溯（reasoning trace） |
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
+-------------------+  +----------------+  +----------------------------------+
| Document Pipeline |  | Index Engine   |  |        Agentic RAG Engine       |
| - Parser          |  | (LlamaIndex)   |  |  +------------+  +------------+ |
| - OCR             |  |                |  |  |   Agent    |  |    LLM     | |
| - Normalizer      |  |                |  |  | Controller |  |  Service   | |
+-------------------+  +----------------+  |  +------------+  +------------+ |
          |                    |           |        |                |       |
          |                    |           |        v                v       |
          |                    |           |  +------------+  +------------+ |
          |                    |           |  |   Tools    |  | Feedback   | |
          |                    |           |  | (Context/  |  | Evaluator  | |
          |                    |           |  |  Analysis) |  |            | |
          |                    |           |  +------------+  +------------+ |
          |                    |           +----------------------------------+
          |                    |                    |
          v                    v                    v
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
│   │   ├── normalizer_service.py # 结构规范化服务
│   │   ├── index_service.py      # 索引构建服务
│   │   ├── topic_service.py      # 话题提取与聚类
│   │   ├── qa_service.py         # 问答入口服务
│   │   └── llm_service.py        # LLM/Embedding 调用封装
│   ├── agent/                # Agentic RAG 引擎
│   │   ├── controller.py     # Agent 控制器（推理循环）
│   │   ├── tools/            # Agent 工具集
│   │   │   ├── context_tools.py   # 上下文增强工具
│   │   │   ├── analysis_tools.py  # 分析工具
│   │   │   └── external_tools.py  # 外部行动工具（扩展）
│   │   ├── evaluator.py      # 反馈评估器
│   │   ├── strategy.py       # 自适应策略
│   │   └── prompts.py        # Agent 提示词模板
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
| depth | INTEGER | 层级深度（document=0, H1=1, H2=2, ...） |
| title | VARCHAR(500) | 标题（章节名） |
| content | TEXT | 文本内容 |
| summary | TEXT | 节点摘要（section 层自动生成） |
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

#### 4.3.1 自适应层级设计

文档解析后形成**自适应深度**的树状结构，使用 `depth` 字段标识层级，不限制最大层数：

```
Document (depth=0)
├── Section: "第1章 概述" (depth=1)
│   ├── Paragraph: "概述介绍文字..." (depth=2, position=0)
│   ├── Section: "1.1 背景" (depth=2, position=1)
│   │   ├── Section: "1.1.1 行业现状" (depth=3)
│   │   │   ├── Paragraph: "当前行业..." (depth=4)
│   │   │   └── Image: [OCR + 描述] (depth=4)
│   │   └── Section: "1.1.2 技术趋势" (depth=3)
│   └── Section: "1.2 目标" (depth=2, position=2)
├── Section: "第2章 详细设计" (depth=1)
│   └── ...
```

**设计要点**：
- **paragraph 和 section 可同级**：用 `position` 保持原始文档顺序
- **depth 自适应**：根据文档实际标题层级确定，无硬性上限
- **每个 section 有 summary**：入库时用 LLM 自动生成

#### 4.3.2 上下文增强（Breadcrumb + Summary）

检索到 paragraph 时，向上收集所有祖先节点的标题和摘要：

```json
{
  "content": "预计投入 500 万预算用于容器化改造...",
  "breadcrumb": [
    {
      "title": "第3章 基础设施",
      "summary": "本章介绍 IT 基础设施升级规划，包括云平台、网络、安全",
      "depth": 1
    },
    {
      "title": "3.2 云平台建设",
      "summary": "计划将核心业务迁移至混合云架构，提升弹性和成本效益",
      "depth": 2
    },
    {
      "title": "3.2.1 容器化改造",
      "summary": "采用 Kubernetes 对现有应用进行容器化改造",
      "depth": 3
    }
  ],
  "document": "2024年度技术规划.docx"
}
```

#### 4.3.3 检索策略

- **检索单位**：仅检索 paragraph 节点（叶子节点）
- **上下文增强**：递归收集祖先路径的 title + summary
- **引用定位**：`文档名 > 章节路径`（如 `第3章 > 3.2 > 3.2.1`）

---

## 5. 核心算法设计

### 5.1 文档解析流程

```
+-------------+     +-------------------+     +------------------+
| Upload File | --> | Detect Type       | --> | Has Structure?   |
+-------------+     +-------------------+     +------------------+
                                                      |
                          +---------------------------+---------------------------+
                          |                                                       |
                          v                                                       v
                 +------------------+                                   +------------------+
                 | Structured Doc   |                                   | Unstructured Doc |
                 | (has headings)   |                                   | (plain text)     |
                 +------------------+                                   +------------------+
                          |                                                       |
                          v                                                       v
                 +------------------+                                   +------------------+
                 | Parse by         |                                   | Semantic         |
                 | Heading Styles   |                                   | Chunking         |
                 +------------------+                                   +------------------+
                          |                                                       |
                          |                                                       v
                          |                                             +------------------+
                          |                                             | LLM: Generate    |
                          |                                             | Titles           |
                          |                                             +------------------+
                          |                                                       |
                          +---------------------------+---------------------------+
                                                      |
                                                      v
                                             +------------------+
                                             | Build Node Tree  |
                                             +------------------+
                                                      |
                          +---------------------------+---------------------------+
                          |                                                       |
                          v                                                       v
                 +------------------+                                   +------------------+
                 | Extract Text     |                                   | Extract Images   |
                 +------------------+                                   +------------------+
                          |                                                       |
                          v                                                       v
                 +------------------+                                   +------------------+
                 | Generate         |                                   | OCR + Multimodal |
                 | Embeddings       |                                   | (PaddleOCR+LLM)  |
                 +------------------+                                   +------------------+
                          |                                                       |
                          +---------------------------+---------------------------+
                                                      |
                                                      v
                                             +------------------+
                                             | Generate Section |
                                             | Summaries        |
                                             | (bottom-up)      |
                                             +------------------+
                                                      |
                                                      v
                                             +------------------+
                                             | Store to DB      |
                                             +------------------+
```

#### 5.1.1 结构化文档解析

各格式层级识别方式：

| 格式 | 层级识别方式 | 图片处理 |
|------|--------------|----------|
| DOCX | 标题样式（Heading 1/2/3/...） | 内嵌图片提取 |
| PPTX | 每页为一个 Section | 幻灯片图片/图表 |
| XLSX | 每个 Sheet 为一个 Section | 图表转图片处理 |
| Markdown | 标题层级（#/##/###/...） | 引用图片路径 |

#### 5.1.2 无结构文档自动结构化

对于没有标题的纯文本文档（如长文章、聊天记录、日志），自动构建层级结构：

**Step 1: 语义切分**

使用滑动窗口 + 嵌入相似度检测"话题转换点"：

```
function semantic_chunking(text):
    sentences = split_into_sentences(text)
    embeddings = embed_batch(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        # Calculate similarity between adjacent sentences
        similarity = cosine_similarity(embeddings[i], embeddings[i-1])

        if similarity < THRESHOLD:  # Topic shift detected
            chunks.append(join(current_chunk))
            current_chunk = []

        current_chunk.append(sentences[i])

    chunks.append(join(current_chunk))
    return chunks
```

**Step 2: LLM 生成标题**

对每个 chunk 生成标题和摘要：

```
function generate_structure(chunks):
    sections = []
    for chunk in chunks:
        prompt = """
        Generate a short title and summary for this text section.
        Return JSON: {"title": "...", "summary": "..."}

        Text:
        {chunk}
        """
        result = llm.generate(prompt)
        sections.append({
            "title": result.title,
            "summary": result.summary,
            "content": chunk
        })
    return sections
```

**Step 3: 层级聚合（可选）**

如果 chunks 太多（>20 个），二次聚类形成更高层级：

```
function hierarchical_grouping(sections):
    if len(sections) <= 20:
        return sections

    # Group similar sections
    embeddings = embed_batch([s.title for s in sections])
    clusters = hierarchical_clustering(embeddings, n_clusters=5)

    # Generate parent section for each cluster
    grouped = []
    for cluster in clusters:
        parent_title = llm.generate_parent_title(cluster)
        grouped.append({
            "title": parent_title,
            "children": cluster
        })
    return grouped
```

### 5.1.3 结构规范化

现实文档经常存在"有结构但结构不合理"的情况，需要在解析后进行规范化处理：

| 问题场景 | 处理方式 |
|----------|----------|
| Section 过长（无子标题） | 语义切分，自动创建子 section |
| Paragraph 过长 | 语义切分为多个 paragraph |
| 标题无意义（如"一、二、三"） | LLM 生成有意义的标题补充 |

#### 规范化流程

```
+------------------+     +------------------+     +------------------+
| Parse Original   | --> | Structure        | --> | Build Final      |
| Structure        |     | Normalization    |     | Node Tree        |
+------------------+     +------------------+     +------------------+
```

#### 规范化逻辑

```
function normalize_structure(node):
    # Rule 1: Split long paragraphs
    if node.type == "paragraph" and len(node.content) > MAX_PARA_LENGTH:
        chunks = semantic_split(node.content, target_size=PARA_TARGET_SIZE)
        replace_with_multiple_paragraphs(node, chunks)

    # Rule 2: Split long sections without subsections
    if node.type == "section" and not has_subsections(node):
        total_content = get_total_content_length(node)
        if total_content > MAX_SECTION_LENGTH:
            subsections = create_subsections(node)
            replace_children(node, subsections)

    # Rule 3: Enrich meaningless titles
    if node.type == "section" and is_meaningless_title(node.title):
        node.title = generate_title_from_content(node)

    # Recurse
    for child in node.children:
        normalize_structure(child)
```

#### 长段落语义切分

在目标长度附近寻找语义断点（相邻句子相似度最低处）：

```
function semantic_split(text, target_size):
    if len(text) <= target_size * 1.2:
        return [text]

    sentences = split_into_sentences(text)
    embeddings = embed_batch(sentences)

    # Find best split point near target_size
    current_length = 0
    best_split = None
    min_similarity = 1.0

    for i, sent in enumerate(sentences):
        current_length += len(sent)

        if current_length >= target_size * 0.8:
            if i < len(sentences) - 1:
                sim = cosine_similarity(embeddings[i], embeddings[i+1])
                if sim < min_similarity:
                    min_similarity = sim
                    best_split = i + 1

        if current_length >= target_size * 1.2:
            break

    if best_split:
        part1 = join(sentences[:best_split])
        part2 = join(sentences[best_split:])
        return [part1] + semantic_split(part2, target_size)
    else:
        return [text]
```

#### 长 Section 自动拆分

```
function create_subsections(section_node):
    paragraphs = get_all_paragraphs(section_node)
    all_content = [p.content for p in paragraphs]

    # Semantic chunking
    chunks = semantic_chunking(join(all_content))

    # Generate titles for each chunk
    subsections = []
    for chunk in chunks:
        title = llm.generate_title(chunk)
        subsections.append({
            "type": "section",
            "title": title,
            "content": chunk,
            "auto_generated": True  # Mark as auto-generated
        })

    return subsections
```

#### 无意义标题检测与增强

```
function is_meaningless_title(title):
    patterns = [
        r'^[一二三四五六七八九十]+[、.]?\s*$',     # 一、二、三
        r'^[0-9]+[、.]\s*$',                       # 1. 2. 3.
        r'^第[一二三四五六七八九十\d]+[章节部分]$', # 第一章（无后缀）
        r'^[A-Z][、.]\s*$',                        # A. B. C.
    ]
    return any(re.match(p, title.strip()) for p in patterns)


function generate_title_from_content(node):
    content_preview = get_content_preview(node, max_length=500)

    prompt = f"""
    This section has a meaningless title "{node.title}".
    Generate a meaningful title (5-15 chars) based on content.

    Content: {content_preview}

    Return: {{"title": "..."}}
    """
    result = llm.generate(prompt)
    return f"{node.title} {result.title}"  # Keep original numbering
```

### 5.2 层级索引构建

#### 5.2.1 节点树构建

索引构建伪代码：

```
function build_index(document):
    # 1. Detect document structure
    if has_headings(document):
        sections = parse_by_headings(document)
    else:
        sections = auto_structure(document)  # Semantic chunking + LLM

    # 2. Build node tree with depth
    root_node = create_node(
        type="document",
        title=document.name,
        depth=0
    )

    build_tree_recursive(root_node, sections, depth=1)

    # 3. Process images
    for image in document.images:
        process_image(image, find_parent_node(image))

    # 4. Generate embeddings for paragraphs
    for para_node in get_all_paragraphs(root_node):
        embedding = embed(para_node.content)
        store_embedding(para_node, embedding)

    # 5. Generate summaries (bottom-up)
    generate_summaries_recursive(root_node)

    # 6. Store to database
    save_nodes_to_db(root_node)


function build_tree_recursive(parent, sections, depth):
    for i, section in enumerate(sections):
        if section.type == "heading":
            section_node = create_node(
                type="section",
                title=section.title,
                depth=depth,
                position=i,
                parent=parent
            )
            # Recursively process children
            build_tree_recursive(section_node, section.children, depth + 1)
        else:
            # Paragraph or text block
            para_node = create_node(
                type="paragraph",
                content=section.text,
                depth=depth,
                position=i,
                parent=parent
            )
```

#### 5.2.2 Summary 生成（自底向上）

Section 的摘要通过 LLM 自动生成，采用自底向上策略：

```
function generate_summaries_recursive(node):
    if node.type == "paragraph":
        # Paragraph: no summary needed, content itself is the summary
        return node.content

    # Collect content from all children
    children_content = []
    for child in node.children:
        child_summary = generate_summaries_recursive(child)
        children_content.append(child_summary)

    # Generate summary using LLM
    if node.type == "section":
        prompt = f"""
        Summarize the following section content in 1-2 sentences.
        Section title: {node.title}

        Content:
        {join(children_content, "\n")}
        """
        node.summary = llm.generate(prompt, max_tokens=100)

    elif node.type == "document":
        prompt = f"""
        Summarize the following document in 2-3 sentences.
        Document: {node.title}

        Sections:
        {join([f"- {c.title}: {c.summary}" for c in node.children], "\n")}
        """
        node.summary = llm.generate(prompt, max_tokens=150)

    return node.summary
```

**Summary 生成时机**：文档入库时一次性生成，后续检索直接使用。

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

### 5.4 版本管理

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

## 6. Agentic RAG 系统

本章描述问答系统的核心推理引擎，采用 Agent 架构实现自适应的文档检索与问答。

### 6.1 设计理念

#### 6.1.1 Agent 三要素

基于 AI Agent 的核心理论，本系统的 Agent 由三个要素构成：

```
+------------------------------------------------------------------+
|                         Agent 系统                                |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------+                                             |
|  |     大模型       |  决策引擎：理解问题、规划行动、生成答案      |
|  | (LLM as Brain)   |                                             |
|  +------------------+                                             |
|           |                                                       |
|           v                                                       |
|  +------------------+                                             |
|  |     上下文       |  动态记忆：问题、已检索信息、推理历史        |
|  | (Context/Memory) |                                             |
|  +------------------+                                             |
|           |                                                       |
|           v                                                       |
|  +------------------+                                             |
|  |      工具        |  行动能力：检索、浏览、分析、验证            |
|  |    (Tools)       |                                             |
|  +------------------+                                             |
|                                                                   |
+------------------------------------------------------------------+
```

#### 6.1.2 核心设计原则

| 原则 | 说明 |
|------|------|
| **先快后慢** | 简单问题直接 RAG 回答，复杂问题才启动 Agent |
| **上下文为王** | 工具的核心目的是提升上下文的信息密度和质量 |
| **反馈驱动** | 每次工具调用后评估效果，动态调整策略 |
| **自主决策** | Agent 自主判断何时检索、何时停止、何时回答 |

### 6.2 工具体系

工具分为三类，服务于不同目的：

#### 6.2.1 工具分类

```
Agent 工具体系
├── 上下文增强工具（Context Enhancement）
│   │  目的：获取更多相关信息，扩充上下文
│   ├── search_documents      # 语义检索文档
│   ├── get_node_content      # 获取节点完整内容
│   ├── get_section_children  # 浏览章节子节点
│   ├── get_document_structure# 获取文档结构树
│   └── get_related_documents # 获取相关文档
│
├── 分析工具（Analysis）
│   │  目的：处理已有信息，提升质量
│   ├── verify_consistency    # 验证信息一致性
│   └── summarize_findings    # 汇总已收集信息
│
└── 外部行动工具（External Action）- 扩展
    │  目的：突破内部知识边界
    └── search_web            # 外部搜索（未来）
```

#### 6.2.2 上下文增强工具

这类工具的目的是**扩充和丰富上下文**，为 LLM 提供更多决策依据：

```python
context_tools = [
    {
        "name": "search_documents",
        "category": "context_enhancement",
        "description": "Search internal documents by semantic similarity. Use when you need to find information on a specific topic.",
        "parameters": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "default": 5, "description": "Number of results"}
        },
        "returns": "List of relevant document paragraphs with metadata"
    },
    {
        "name": "get_node_content",
        "category": "context_enhancement",
        "description": "Get the full content of a specific node. Use when a search result snippet is not enough and you need more details.",
        "parameters": {
            "node_id": {"type": "string", "description": "Node ID from search results"}
        },
        "returns": "Full node content with parent context"
    },
    {
        "name": "get_section_children",
        "category": "context_enhancement",
        "description": "Get all child nodes of a section. Use when you want to explore a topic in depth.",
        "parameters": {
            "node_id": {"type": "string", "description": "Section node ID"}
        },
        "returns": "List of child nodes (subsections and paragraphs)"
    },
    {
        "name": "get_document_structure",
        "category": "context_enhancement",
        "description": "Get the hierarchical structure of a document. Use when you need to understand how a document is organized.",
        "parameters": {
            "document_id": {"type": "string", "description": "Document ID"}
        },
        "returns": "Document tree structure with section titles and summaries"
    },
    {
        "name": "get_related_documents",
        "category": "context_enhancement",
        "description": "Get documents related by shared topics. Use for multi-hop reasoning when information might span multiple documents.",
        "parameters": {
            "document_id": {"type": "string", "description": "Document ID"}
        },
        "returns": "List of related documents with relevance scores"
    }
]
```

#### 6.2.3 分析工具

这类工具的目的是**处理和验证已有信息**，提升上下文质量：

```python
analysis_tools = [
    {
        "name": "verify_consistency",
        "category": "analysis",
        "description": "Verify if gathered information is consistent and non-contradictory. Use before final answer to ensure quality.",
        "parameters": {
            "statements": {"type": "array", "description": "List of statements to verify"}
        },
        "returns": "Consistency analysis with any contradictions identified"
    },
    {
        "name": "summarize_findings",
        "category": "analysis",
        "description": "Summarize all information gathered so far. Use when context is getting large and needs consolidation.",
        "parameters": {
            "focus": {"type": "string", "description": "Aspect to focus the summary on"}
        },
        "returns": "Consolidated summary of findings"
    }
]
```

#### 6.2.4 外部行动工具（扩展）

```python
external_tools = [
    {
        "name": "search_web",
        "category": "external_action",
        "description": "Search the web for information not in internal documents. Use as fallback when internal search yields insufficient results.",
        "parameters": {
            "query": {"type": "string", "description": "Search query"}
        },
        "returns": "Web search results",
        "enabled": False  # Future extension
    }
]
```

#### 6.2.5 终止工具

```python
terminal_tools = [
    {
        "name": "final_answer",
        "category": "terminal",
        "description": "Provide the final answer. Use when you have gathered sufficient information to answer the question.",
        "parameters": {
            "answer": {"type": "string", "description": "The answer with citations"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "sources": {"type": "array", "description": "List of sources used"}
        }
    }
]
```

### 6.3 反馈与自适应

Agent 的核心能力是**根据反馈动态调整策略**，而非机械执行。

#### 6.3.1 反馈循环架构

```
+----------+     +----------+     +----------+     +----------+
|  Think   | --> |   Act    | --> | Observe  | --> | Evaluate |
| (Reason) |     | (Tool)   |     | (Result) |     | (Feedback|
+----------+     +----------+     +----------+     +----------+
     ^                                                   |
     |                                                   |
     +---------------------------------------------------+
                    Feedback Loop
```

#### 6.3.2 反馈评估机制

每次工具调用后，评估结果并更新状态：

```python
def evaluate_tool_result(action, result, context):
    """Evaluate the result of a tool call and update context state."""

    evaluation = {
        "action": action,
        "result_quality": None,
        "context_improvement": None,
        "next_suggestion": None
    }

    # 1. Assess result quality
    if action.tool == "search_documents":
        if len(result.items) == 0:
            evaluation["result_quality"] = "empty"
            evaluation["next_suggestion"] = "try_different_query"
        elif max(r.score for r in result.items) < 0.5:
            evaluation["result_quality"] = "low_relevance"
            evaluation["next_suggestion"] = "refine_query_or_explore"
        else:
            evaluation["result_quality"] = "good"
            evaluation["context_improvement"] = "expanded"

    elif action.tool == "get_node_content":
        if result.content:
            evaluation["result_quality"] = "good"
            evaluation["context_improvement"] = "deepened"
        else:
            evaluation["result_quality"] = "empty"
            evaluation["next_suggestion"] = "try_related_node"

    elif action.tool == "verify_consistency":
        if result.has_contradictions:
            evaluation["result_quality"] = "contradictions_found"
            evaluation["next_suggestion"] = "resolve_contradictions"
        else:
            evaluation["result_quality"] = "consistent"
            evaluation["next_suggestion"] = "ready_to_answer"

    # 2. Update context state
    context["evaluations"].append(evaluation)
    context["info_density"] = calculate_info_density(context)

    return evaluation
```

#### 6.3.3 自适应策略

根据反馈调整后续行动：

```python
def adaptive_strategy(context, evaluation):
    """Determine next action based on feedback."""

    # Strategy 1: Query refinement
    if evaluation["next_suggestion"] == "try_different_query":
        return {
            "strategy": "refine_query",
            "hint": "Try synonyms or related terms"
        }

    # Strategy 2: Depth exploration
    if evaluation["next_suggestion"] == "refine_query_or_explore":
        if context["depth_explored"] < 2:
            return {
                "strategy": "explore_deeper",
                "hint": "Get more details from existing results"
            }
        else:
            return {
                "strategy": "broaden_search",
                "hint": "Search with broader terms"
            }

    # Strategy 3: Contradiction resolution
    if evaluation["next_suggestion"] == "resolve_contradictions":
        return {
            "strategy": "verify_sources",
            "hint": "Check original documents for accuracy"
        }

    # Strategy 4: Ready to answer
    if context["info_density"] >= SUFFICIENT_THRESHOLD:
        return {
            "strategy": "finalize",
            "hint": "Sufficient information gathered"
        }

    return {"strategy": "continue", "hint": None}
```

### 6.4 执行框架

#### 6.4.1 整体流程

```
+----------+     +------------------+     +-------------------+
| Question | --> | Standard RAG     | --> | Quality Check     |
|          |     | (single search)  |     | (sufficient?)     |
+----------+     +------------------+     +-------------------+
                                                   |
                          +------------------------+------------------------+
                          | Yes                                             | No
                          v                                                 v
                 +------------------+                              +------------------+
                 | Direct Answer    |                              | Agent Loop       |
                 | (fast path)      |                              | (deep reasoning) |
                 +------------------+                              +------------------+
                          |                                                 |
                          |                                        +--------+--------+
                          |                                        |                 |
                          |                                        v                 v
                          |                               +-------------+    +-------------+
                          |                               |   Think     |    |  Evaluate   |
                          |                               | (plan next) |<-->| (feedback)  |
                          |                               +-------------+    +-------------+
                          |                                        |
                          |                                        v
                          |                               +-------------+
                          |                               |    Act      |
                          |                               | (use tool)  |
                          |                               +-------------+
                          |                                        |
                          +------------------------+---------------+
                                                   |
                                                   v
                                          +------------------+
                                          | Answer + Sources |
                                          +------------------+
```

#### 6.4.2 主入口函数

```python
def answer_question(question: str, version: str = None) -> dict:
    """Main entry point for question answering."""

    # Step 1: Standard RAG retrieval (fast path attempt)
    results, contexts = standard_rag(question, version)

    # Step 2: Check if retrieval is sufficient
    quality = check_retrieval_quality(question, results, contexts)

    if quality["sufficient"]:
        # Fast path: Direct answer
        answer = generate_direct_answer(question, contexts)
        return {
            "answer": answer,
            "sources": format_citations(contexts),
            "mode": "direct",
            "reasoning_steps": 1
        }
    else:
        # Slow path: Agent reasoning
        result = agent_reasoning_loop(question, contexts, quality["reason"])
        result["mode"] = "agent"
        return result
```

#### 6.4.3 Agent 推理循环

```python
def agent_reasoning_loop(question: str, initial_contexts: list,
                         initial_reason: str, max_iterations: int = 3) -> dict:
    """Agent reasoning loop with feedback."""

    # Initialize context state
    context = {
        "question": question,
        "gathered_info": initial_contexts,
        "evaluations": [],
        "reasoning_trace": [],
        "info_density": calculate_info_density(initial_contexts),
        "depth_explored": 0,
        "iteration": 0
    }

    # Add initial assessment
    context["reasoning_trace"].append({
        "step": 0,
        "type": "initial_assessment",
        "content": f"Standard RAG insufficient: {initial_reason}"
    })

    for i in range(max_iterations):
        context["iteration"] = i + 1

        # 1. Think: Plan next action
        thought, action = agent_think(context)
        context["reasoning_trace"].append({
            "step": i + 1,
            "type": "thought",
            "content": thought
        })

        # 2. Check for terminal action
        if action.tool == "final_answer":
            return {
                "answer": action.params["answer"],
                "confidence": action.params.get("confidence", "medium"),
                "sources": action.params["sources"],
                "reasoning_trace": context["reasoning_trace"],
                "iterations": i + 1
            }

        # 3. Act: Execute tool
        result = execute_tool(action.tool, action.params)
        context["reasoning_trace"].append({
            "step": i + 1,
            "type": "action",
            "tool": action.tool,
            "params": action.params,
            "result_summary": summarize_result(result)
        })

        # 4. Observe & Evaluate: Assess result quality
        evaluation = evaluate_tool_result(action, result, context)

        # 5. Update context with new information
        update_context(context, action, result, evaluation)

        # 6. Adaptive strategy adjustment
        strategy = adaptive_strategy(context, evaluation)
        context["current_strategy"] = strategy

        # 7. Early termination if sufficient
        if strategy["strategy"] == "finalize":
            # Run verification before final answer
            verification = verify_before_answer(context)
            if verification["ready"]:
                return synthesize_final_answer(context)

    # Max iterations reached
    return synthesize_final_answer(context, forced=True)
```

#### 6.4.4 Agent 思考提示词

```python
def agent_think(context: dict) -> tuple:
    """Generate next thought and action."""

    history = format_reasoning_history(context["reasoning_trace"])
    strategy_hint = context.get("current_strategy", {}).get("hint", "")

    prompt = f"""You are a research assistant answering questions by exploring documents.

## Question
{context["question"]}

## Available Tools

**Context Enhancement:**
- search_documents(query, top_k): Search for relevant documents
- get_node_content(node_id): Get full content of a node
- get_section_children(node_id): Explore section in depth
- get_document_structure(document_id): Understand document organization
- get_related_documents(document_id): Find connected documents

**Analysis:**
- verify_consistency(statements): Check for contradictions
- summarize_findings(focus): Consolidate gathered information

**Terminal:**
- final_answer(answer, confidence, sources): Provide final answer

## Reasoning History
{history}

## Current State
- Iteration: {context["iteration"]} / {MAX_ITERATIONS}
- Information density: {context["info_density"]:.2f}
- Strategy hint: {strategy_hint or "None"}

## Instructions
Think step by step:
1. What do I know so far? Is it sufficient?
2. What is missing or unclear?
3. Which tool will best fill the gap?
4. Am I ready to provide final_answer?

Respond in this format:
Thought: <your reasoning>
Action: <tool_name>
Parameters: <json>
"""

    response = llm.generate(prompt)
    return parse_agent_response(response)
```

### 6.5 标准 RAG 流程

当问题简单时，走快速路径，无需 Agent 循环。

#### 6.5.1 标准检索

```python
def standard_rag(question: str, version: str = None) -> tuple:
    """Standard RAG retrieval."""

    # 1. Embed question
    query_embedding = embed(question)

    # 2. Identify relevant topics
    topics = identify_question_topic(question)

    # 3. Vector search
    results = vector_search(
        table="embeddings",
        query=query_embedding,
        top_k=10,
        filter={
            "version": version or "latest",
            "topics": topics
        }
    )

    # 4. Build contexts with breadcrumbs
    contexts = []
    for result in results:
        node = get_node(result.node_id)
        contexts.append({
            "content": node.content,
            "breadcrumb": build_breadcrumb(node),
            "document_name": get_document_name(node),
            "relevance": result.score,
            "node_id": result.node_id
        })

    return results, contexts
```

#### 6.5.2 质量检查

```python
def check_retrieval_quality(question: str, results: list, contexts: list) -> dict:
    """Check if retrieval results are sufficient."""

    # Check 1: Basic relevance threshold
    if len(results) == 0:
        return {"sufficient": False, "reason": "no_results"}

    max_score = max(r.score for r in results)
    if max_score < 0.7:
        return {"sufficient": False, "reason": "low_relevance"}

    # Check 2: LLM judgment
    judgment = llm.generate(f"""
    Question: {question}

    Retrieved content:
    {format_contexts(contexts[:5])}

    Can this content directly answer the question?
    Return JSON: {{"sufficient": true/false, "reason": "explanation"}}
    """)

    return json.loads(judgment)
```

#### 6.5.3 Breadcrumb 构建

```python
def build_breadcrumb(node) -> list:
    """Build breadcrumb with ancestor titles and summaries."""

    breadcrumb = []
    current = node.parent

    while current and current.type != "document":
        breadcrumb.insert(0, {
            "title": current.title,
            "summary": current.summary,
            "depth": current.depth
        })
        current = current.parent

    return breadcrumb
```

### 6.6 响应格式

#### 6.6.1 直接回答模式

```json
{
  "answer": "根据文档记载，项目的主要目标是...",
  "sources": [
    {
      "document": "项目规划.docx",
      "path": "第1章 概述 > 1.2 项目目标",
      "snippet": "...核心目标是建立一个统一的数据平台...",
      "relevance": 0.92
    }
  ],
  "mode": "direct",
  "reasoning_steps": 1
}
```

#### 6.6.2 Agent 推理模式

```json
{
  "answer": "根据多个文档的综合分析，A项目负责人张三此前还负责过B项目和C项目...",
  "confidence": "high",
  "sources": [
    {
      "document": "A项目规划.docx",
      "path": "第1章 > 1.1 项目团队",
      "snippet": "项目负责人：张三..."
    },
    {
      "document": "B项目总结.docx",
      "path": "封面",
      "snippet": "项目经理：张三..."
    }
  ],
  "mode": "agent",
  "iterations": 2,
  "reasoning_trace": [
    {"step": 0, "type": "initial_assessment", "content": "需要多跳推理"},
    {"step": 1, "type": "thought", "content": "先找A项目负责人是谁"},
    {"step": 1, "type": "action", "tool": "search_documents", "result_summary": "找到张三"},
    {"step": 2, "type": "thought", "content": "用张三搜索其他项目"},
    {"step": 2, "type": "action", "tool": "search_documents", "result_summary": "找到B、C项目"}
  ]
}
```

---

## 7. 接口设计

### 7.1 后端 API

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
  "version": null,           // 可选，null 表示最新版本
  "show_reasoning": false    // 可选，是否返回推理过程
}

Response (直接回答模式):
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
  "identified_topics": ["项目管理", "规划"],
  "mode": "direct",
  "reasoning_steps": 1
}

Response (Agent 推理模式):
{
  "answer": "根据多个文档的综合分析，A项目负责人张三此前还负责过B项目...",
  "confidence": "high",
  "sources": [
    {
      "document_id": "uuid1",
      "document_name": "A项目规划.docx",
      "section": "第1章 > 1.1 项目团队",
      "snippet": "项目负责人：张三...",
      "relevance": 0.95
    },
    {
      "document_id": "uuid2",
      "document_name": "B项目总结.docx",
      "section": "封面",
      "snippet": "项目经理：张三...",
      "relevance": 0.88
    }
  ],
  "identified_topics": ["项目管理", "人员"],
  "mode": "agent",
  "iterations": 2,
  "reasoning_trace": [
    {"step": 0, "type": "initial_assessment", "content": "需要多跳推理"},
    {"step": 1, "type": "thought", "content": "先找A项目负责人是谁"},
    {"step": 1, "type": "action", "tool": "search_documents", "result_summary": "找到张三"},
    {"step": 2, "type": "thought", "content": "用张三搜索其他项目"},
    {"step": 2, "type": "action", "tool": "search_documents", "result_summary": "找到B项目"}
  ]
}
```

### 7.2 前端页面

| 页面 | 功能 |
|------|------|
| **首页/仪表盘** | 文档统计、最近上传、热门话题 |
| **文档管理** | 拖拽上传、文档列表、搜索过滤、版本查看 |
| **话题浏览** | 话题卡片/标签云、点击查看关联文档 |
| **问答界面** | 对话式问答、话题/版本筛选、引用来源展示、推理过程展示（可折叠） |
| **文档详情** | 文档结构树、版本历史、关联话题 |

#### 问答界面特性

问答界面需要支持 Agentic RAG 的交互特性：

```
+------------------------------------------------------------------+
|                         问答界面                                   |
+------------------------------------------------------------------+
| +--------------------------------------------------------------+ |
| |  [话题筛选 v]  [版本选择 v]  [显示推理过程 □]                  | |
| +--------------------------------------------------------------+ |
|                                                                   |
| +--------------------------------------------------------------+ |
| | 用户: A项目负责人之前还负责过哪些项目？                        | |
| +--------------------------------------------------------------+ |
|                                                                   |
| +--------------------------------------------------------------+ |
| | 助手:                                                          | |
| | 根据多个文档的综合分析，A项目负责人张三此前还负责过B项目...     | |
| |                                                                | |
| | [高置信度] [Agent模式] [2轮推理]                               | |
| |                                                                | |
| | +----------------------------------------------------------+  | |
| | | ▼ 推理过程                                                |  | |
| | |   Step 1: 搜索A项目负责人信息 → 找到张三                   |  | |
| | |   Step 2: 搜索张三参与的其他项目 → 找到B项目               |  | |
| | +----------------------------------------------------------+  | |
| |                                                                | |
| | 来源:                                                          | |
| | • A项目规划.docx > 第1章 > 1.1 项目团队 (95%)                  | |
| | • B项目总结.docx > 封面 (88%)                                  | |
| +--------------------------------------------------------------+ |
+------------------------------------------------------------------+
```

**交互要点**：
- 推理过程默认折叠，点击可展开
- 回答模式（direct/agent）用标签显示
- 置信度用颜色区分（high=绿色, medium=黄色, low=红色）
- 来源支持点击跳转到文档详情

---

## 8. 测试方案

### 8.1 单元测试

| 模块 | 测试要点 |
|------|----------|
| parser_service | 各格式解析正确性、层级结构提取 |
| normalizer_service | 长段落切分、长 section 拆分、无意义标题增强 |
| index_service | 节点创建、嵌入生成、存储正确性 |
| topic_service | 话题提取、相似度匹配、聚类逻辑 |
| qa_service | 检索准确性、上下文拼接、引用格式 |
| version | 版本创建、版本过滤查询 |
| **agent/controller** | 推理循环控制、迭代终止条件、状态管理 |
| **agent/tools** | 各工具输入输出、错误处理、结果格式 |
| **agent/evaluator** | 结果质量评估、反馈生成准确性 |
| **agent/strategy** | 策略选择逻辑、策略切换条件 |

#### Agent 模块测试详情

```python
# tests/unit/agent/test_controller.py
class TestAgentController:
    def test_simple_question_fast_path(self):
        """简单问题走直接回答路径"""
        pass

    def test_complex_question_triggers_agent(self):
        """复杂问题触发 Agent 循环"""
        pass

    def test_max_iterations_limit(self):
        """达到最大迭代次数时强制终止"""
        pass

    def test_early_termination_on_sufficient_info(self):
        """信息充足时提前终止"""
        pass


# tests/unit/agent/test_tools.py
class TestContextTools:
    def test_search_documents_returns_ranked_results(self):
        """文档搜索返回排序结果"""
        pass

    def test_get_node_content_with_breadcrumb(self):
        """获取节点内容包含面包屑"""
        pass


class TestAnalysisTools:
    def test_verify_consistency_detects_contradictions(self):
        """一致性验证能检测矛盾"""
        pass

    def test_summarize_findings_consolidates_info(self):
        """汇总工具能整合信息"""
        pass


# tests/unit/agent/test_evaluator.py
class TestEvaluator:
    def test_evaluate_empty_search_results(self):
        """空搜索结果的评估"""
        pass

    def test_evaluate_low_relevance_results(self):
        """低相关度结果的评估"""
        pass

    def test_calculate_info_density(self):
        """信息密度计算"""
        pass
```

### 8.2 集成测试

| 场景 | 测试要点 |
|------|----------|
| 文档上传流程 | 上传 → 解析 → 规范化 → 索引 → 话题提取 完整流程 |
| 问答流程（直接） | 简单提问 → 检索 → 质量检查通过 → 直接生成 → 引用 |
| 问答流程（Agent） | 复杂提问 → 检索 → 质量检查不通过 → Agent 循环 → 多轮工具调用 → 生成 |
| 多跳推理 | 跨文档关联问题 → 多次搜索 → 信息整合 → 综合回答 |
| 版本更新 | 上传新版本 → 增量索引 → 版本查询 |

#### Agent 推理流程集成测试

```python
# tests/integration/test_agent_flow.py
class TestAgentReasoningFlow:
    def test_multi_hop_reasoning(self):
        """
        场景：A项目负责人还负责过哪些项目？
        预期流程：
        1. 搜索 A 项目 → 找到负责人张三
        2. 搜索张三 → 找到 B、C 项目
        3. 综合回答
        """
        pass

    def test_contradiction_resolution(self):
        """
        场景：两个文档对同一事实描述不一致
        预期流程：
        1. 检索到矛盾信息
        2. verify_consistency 检测到矛盾
        3. 回退到原文验证
        4. 给出带说明的回答
        """
        pass

    def test_insufficient_info_handling(self):
        """
        场景：问题在文档库中找不到答案
        预期流程：
        1. 多次搜索尝试
        2. 信息密度始终低于阈值
        3. 达到最大迭代次数
        4. 返回低置信度回答或承认不知道
        """
        pass

    def test_feedback_loop_strategy_adjustment(self):
        """
        场景：初次搜索结果不佳，需要调整策略
        预期流程：
        1. 初次搜索 → 低相关度
        2. 评估器建议换词搜索
        3. 调整查询词重新搜索
        4. 获得更好结果
        """
        pass
```

### 8.3 测试数据

准备测试文档集：
- 各格式各 2-3 个文档
- 包含中英文内容
- 包含图片、表格
- 包含明确的层级结构
- **包含跨文档关联信息**（用于测试多跳推理）
- **包含信息矛盾的文档对**（用于测试一致性验证）
- **包含无结构纯文本**（用于测试自动结构化）

---

## 9. 开发计划

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

4. **结构规范化服务**
   - 长段落语义切分
   - 长 Section 自动拆分
   - 无意义标题增强

5. **图片处理**
   - PaddleOCR 集成
   - 多模态理解接口
   - 图片描述生成

6. **索引服务**
   - LlamaIndex 集成
   - 层级节点构建
   - 嵌入生成与存储
   - Section Summary 自动生成

### 第三阶段：标准问答功能

7. **RAG 检索**
   - 向量检索实现
   - Breadcrumb 上下文增强
   - 版本过滤

8. **LLM 问答（快速路径）**
   - LLM 服务封装
   - Prompt 模板
   - 引用格式化
   - 质量检查判断

### 第四阶段：Agentic RAG

9. **Agent 工具集**
   - 上下文增强工具（search_documents, get_node_content 等）
   - 分析工具（verify_consistency, summarize_findings）
   - 终止工具（final_answer）

10. **Agent 控制器**
    - 推理循环实现
    - 状态管理
    - 迭代控制

11. **反馈与策略**
    - 结果评估器
    - 自适应策略
    - 信息密度计算

### 第五阶段：话题功能

12. **话题提取**
    - LLM 提取话题
    - 话题相似度合并

13. **话题聚类**
    - 问题话题识别
    - 话题筛选功能

### 第六阶段：前端开发

14. **基础页面**
    - 布局框架
    - 文档上传组件
    - 文档列表

15. **问答界面**
    - 对话组件
    - 引用展示
    - 话题/版本筛选
    - 推理过程展示（可折叠）
    - 置信度/模式标签

16. **话题浏览**
    - 话题列表
    - 关联文档展示

### 第七阶段：完善

17. **版本管理 UI**
18. **性能优化**
19. **部署配置**

---

## 10. 扩展方向

| 方向 | 说明 |
|------|------|
| **复杂版本管理** | diff 对比、跨版本问答（"这个文档改了什么"） |
| **协作功能** | 多用户、权限管理 |
| **更多格式** | PDF、网页、邮件等 |
| **知识图谱** | 实体抽取、关系构建 |
| **主动推荐** | 基于用户行为推荐相关文档 |
| **定时任务** | 监控文件夹自动同步 |
| **Web 搜索** | 内部文档不足时回退到外部搜索引擎 |

---

## 11. 附录

### 11.1 配置项

```yaml
# config.yaml

# 服务配置
server:
  host: "0.0.0.0"
  port: 8000

# 数据库配置
database:
  host: "localhost"
  port: 5432
  name: "dropqa"
  user: "postgres"
  password: "${DB_PASSWORD}"

# 存储配置
storage:
  documents_path: "./data/documents"
  images_path: "./data/images"

# LLM 配置
llm:
  base_url: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 2000

# Embedding 配置
embedding:
  base_url: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"
  model: "text-embedding-3-small"
  dimensions: 1536

# 文档解析配置
parsing:
  # 支持的文件类型
  supported_types:
    - ".docx"
    - ".pptx"
    - ".xlsx"
    - ".md"
    - ".txt"

  # 结构规范化
  structure_normalization:
    max_paragraph_length: 800       # 超过此长度的段落会被切分
    paragraph_target_size: 400      # 段落切分的目标大小
    max_section_length: 3000        # 无子节点的 section 超过此长度会被拆分
    min_section_length: 100         # 过短的 section 可能合并
    enrich_meaningless_titles: true # 是否增强无意义标题

  # 语义切分阈值
  semantic_chunking:
    similarity_threshold: 0.5       # 相似度低于此值视为话题转换
    min_chunk_size: 100             # 最小 chunk 大小
    max_chunk_size: 1000            # 最大 chunk 大小

# OCR 配置
ocr:
  use_gpu: false
  lang: "ch"  # ch: 中英文, en: 英文

# 检索配置
retrieval:
  top_k: 10                         # 默认检索数量
  relevance_threshold: 0.7          # 相关性阈值

# Agent 配置
agent:
  # 基础配置
  max_iterations: 3                 # Agent 最大迭代次数
  enable_web_search: false          # 是否启用 Web 搜索（扩展功能）

  # 质量检查配置
  quality_check:
    min_relevance_score: 0.7        # 触发 Agent 的最低相关度阈值
    use_llm_judgment: true          # 是否使用 LLM 判断检索质量

  # 信息密度配置
  info_density:
    sufficient_threshold: 0.8       # 信息充足阈值（达到则可终止）
    low_threshold: 0.3              # 信息不足阈值（需要更多搜索）

  # 反馈评估配置
  feedback:
    empty_result_action: "refine_query"      # 空结果时的建议动作
    low_relevance_action: "explore_deeper"   # 低相关度时的建议动作
    contradiction_action: "verify_sources"   # 发现矛盾时的建议动作

  # 策略配置
  strategy:
    max_depth_exploration: 3        # 最大深度探索次数
    query_refinement_attempts: 2    # 查询优化尝试次数
    enable_verification: true       # 是否在回答前进行验证

  # 工具配置
  tools:
    search_top_k: 5                 # 搜索工具默认返回数量
    max_context_tokens: 4000        # 上下文最大 token 数
    enable_summarize: true          # 是否启用汇总工具

# 话题配置
topics:
  max_topics_per_document: 5        # 每个文档最多提取的话题数
  similarity_threshold: 0.85        # 话题合并的相似度阈值
```

### 11.2 pgvector 安装

```bash
# PostgreSQL 需要 pgvector 扩展
# Docker 方式
docker run -d \
  --name postgres-vector \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 11.3 依赖列表（初步）

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

### 11.4 参考资料

- [LlamaIndex 文档](https://docs.llamaindex.ai/)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [unstructured 文档](https://unstructured-io.github.io/unstructured/)
- [PaddleOCR 文档](https://paddlepaddle.github.io/PaddleOCR/)
