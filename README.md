# MG ZS RAG Assistant

基于 LangChain、ChromaDB 和 NVIDIA API 构建的汽车手册智能问答助手。通过检索增强生成（Retrieval-Augmented Generation, RAG），让大语言模型能够根据 MG ZS 汽车手册中的内容回答用户问题，支持语义检索和多轮上下文对话。

## Features

- **Document Loading**：加载汽车手册 HTML 文档，并进行文本提取与预处理。
- **Text Embedding**：使用 NVIDIA Embedding 模型将文档转换为向量。
- **Vector Storage**：使用 ChromaDB 持久化存储文档向量。
- **Semantic Search**：根据用户问题检索相关文档片段。
- **RAG Generation**：结合检索结果和用户问题，调用大语言模型生成回答。
- **Multi-turn Conversation**：结合历史对话改写当前问题，支持上下文相关的连续追问。
- **Prompt Engineering**：通过 Prompt 约束回答内容，减少缺乏文档依据的信息生成。

## Tech Stack

| Technology | Description |
|---|---|
| Python 3.12+ | Programming Language |
| uv | Dependency and Environment Management |
| LangChain | LLM Application Framework |
| LCEL | Chain Composition |
| ChromaDB | Vector Database |
| NVIDIA API | LLM and Embedding Services |
| python-dotenv | Environment Variable Management |

### Models

| Model | Name |
|---|---|
| LLM | `nvidia/nemotron-3.5-lightning-30b-a3b` |
| Embedding | `nvidia/nemotron-3.5-embed-1b` |

## Architecture

### Indexing Pipeline

```text
Automotive Manual (HTML)
          |
          v
    Document Loading
          |
          v
     Text Splitting
          |
          v
   NVIDIA Embeddings
          |
          v
      ChromaDB
```

### Retrieval & Generation

```text
      User Question
             |
             v
     Chat History
             |
             v
   Query Contextualization
             |
             v
    NVIDIA Embeddings
             |
             v
     ChromaDB Retrieval
             |
             v
   Retrieved Documents
             |
             v
      Prompt Template
             |
             v
     NVIDIA Nemotron
             |
             v
       AI Response
```

## Project Structure

```text
mg-zs-rag/
├── data/
│   ├── raw/                  # Original manual documents
│   └── chroma/               # ChromaDB vector storage
│
├── src/
│   └── mg_zs_rag/
│       ├── __init__.py
│       ├── config.py         # Project configuration
│       ├── ingest.py         # Document ingestion pipeline
│       └── chat.py           # RAG chat application
│
├── .env.example              # Environment variable template
├── .gitignore
├── pyproject.toml            # Project dependencies
├── uv.lock                   # Dependency lock file
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- NVIDIA API Key

### 1. Clone the repository

```bash
git clone https://github.com/Wang-Mac/mg-zs-rag.git

cd mg-zs-rag
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Configure your NVIDIA API credentials:

```env
NVIDIA_API_KEY=your_nvidia_api_key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

NVIDIA_LLM_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
NVIDIA_EMBEDDING_MODEL=nvidia/nemotron-3.5-embed-1b
```

### 4. Prepare documents

Place the MG ZS manual HTML document in:

```text
data/raw/
```

The original manual is not included in this repository. Please provide your own legally obtained document.

### 5. Build the vector database

```bash
uv run python -m mg_zs_rag.ingest
```

This command loads the document, splits the text, generates embeddings and stores the vectors in ChromaDB.

### 6. Start the application

```bash
uv run python -m mg_zs_rag.chat
```

You can now ask questions about the MG ZS manual through the terminal.

Type `quit` to exit the application.

## Example

**User:**

```text
机油压力过低警告灯亮起是什么意思？
```

**Assistant:**

```text
根据检索到的 MG ZS 车主手册内容：

机油压力过低意味着发动机润滑系统可能无法
提供足够的机油压力，可能导致严重发动机损坏。

手册建议：
1. 在安全允许的情况下停车。
2. 关闭发动机。
3. 检查发动机机油液位。
4. 尽快联系 MG 授权维修厂。
```

**Multi-turn Conversation**

```text
User: 机油压力过低警告灯亮起是什么意思？

Assistant: 机油压力过低意味着可能导致严重发动机损坏……

User: 那我应该怎么处理？

Assistant: 根据前面的机油压力警告，
          应在安全允许的情况下停车并关闭发动机……
```

## Notes

- The current document ingestion pipeline uses HTML documents.
- The vector database is generated locally and is not included in the repository.
- NVIDIA API credentials must be configured locally through `.env`.
- Answers depend on the content retrieved from the provided manual. The system is intended for learning and information retrieval, not as a substitute for professional vehicle diagnosis or repair advice.

## License

This project is intended for learning and technical exploration. A formal open-source license has not yet been selected. Third-party documents and materials remain subject to their original copyright terms.