# Qwen3-Reranker-8B on Replicate

A Cog-powered implementation of Alibaba's Qwen3-Reranker-8B model for text reranking, ready to deploy on Replicate.

## Model Overview

Qwen3-Reranker-8B is a state-of-the-art text reranking model that:
- Supports 100+ languages
- Handles up to 32k context length
- Uses 8B parameters for sophisticated ranking
- Outperforms many existing rerankers

## Features

- **GPU Accelerated**: Uses CUDA for efficient inference
- **Batch Processing**: Handles multiple documents in batches
- **Flexible Input**: Accepts JSON array of documents
- **Top-k Selection**: Returns most relevant documents
- **Optimized Setup**: Uses uv for dependency management

## Setup

The project has been initialized with `uv` for Python package management and configured with Cog for Replicate deployment.

### Dependencies

Core dependencies managed with uv:
- `torch>=2.9.1`
- `transformers>=4.57.1`
- `accelerate>=1.11.0`
- All required CUDA libraries

## Usage

### Local Testing

```bash
# Test the model locally
python test_predict.py
```

### Using the API

The model accepts the following inputs:

- `instruction` (string): Task instruction for the reranker (recommended for better performance)
- `query` (string): The search query or reference text
- `documents` (list): List of documents to rank
- `top_k` (integer, default=5): Number of top results to return
- `batch_size` (integer, default=8): Processing batch size

Example input:
```json
{
  "instruction": "Given a search query and a document, evaluate how relevant the document is to answering the query.",
  "query": "What is artificial intelligence?",
  "documents": ["Document 1 text", "Document 2 text", "Document 3 text"],
  "top_k": 3,
  "batch_size": 2
}
```

The model returns JSON with ranked documents and scores:
```json
{
  "instruction": "Given a search query and a document, evaluate how relevant the document is to answering the query.",
  "query": "What is artificial intelligence?",
  "results": [
    {
      "document": "Most relevant document text...",
      "score": 8.2341
    },
    {
      "document": "Second most relevant document...",
      "score": 7.8912
    }
  ],
  "total_documents": 3
}
```

## Deployment to Replicate

### Prerequisites
- Docker installed and running
- Replicate account
- Cog CLI (`cog`)

### Build and Deploy

1. **Login to Replicate**:
```bash
cog login
```
This will open a browser for authentication.

2. **Build the model**:
```bash
cog build
```

3. **Push to Replicate**:
```bash
cog push r8.im/your-username/qwen3-reranker-8b
```

4. **Create model on Replicate**:
- Go to [Replicate](https://replicate.com)
- Create a new model
- Link it to the pushed Docker image

## File Structure

```
qwen3-reranker-8b/
├── cog.yaml          # Cog configuration
├── predict.py        # Model inference logic
├── requirements.txt  # Python dependencies (uv-managed)
├── test_predict.py   # Local testing script
├── README.md         # This file
└── pyproject.toml    # uv project configuration
```

## Development

### Adding Dependencies

```bash
uv add new-package
uv pip freeze > requirements.txt
```

### Testing

```bash
python test_predict.py
```

### Building

```bash
cog build
```

## Performance Notes

- Model uses `float16` precision for memory efficiency
- Automatic device mapping (`device_map="auto"`)
- Batch processing for optimal throughput
- GPU memory optimization with gradient checkpointing disabled

## License

This model uses the Qwen3-Reranker-8B which is licensed under Apache 2.0.

## References

- [Qwen3-Reranker-8B on Hugging Face](https://huggingface.co/Qwen/Qwen3-Reranker-8B)
- [Cog Documentation](https://cog.run/)
- [Replicate Documentation](https://replicate.com/docs)
