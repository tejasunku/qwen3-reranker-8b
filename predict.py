# Prediction interface for Cog ⚙️
# https://cog.run/python

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from cog import BasePredictor, Input
import json
from typing import List


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the Qwen3-Reranker-8B model into memory to make running multiple predictions efficient"""
        # Try to load from local directory first, then fallback to HuggingFace
        local_model_path = "./model_weights"

        import os
        if os.path.exists(local_model_path) and os.listdir(local_model_path):
            print(f"Loading model from local directory: {local_model_path}")
            model_path = local_model_path
        else:
            print("Loading model from HuggingFace...")
            model_path = "Qwen/Qwen3-Reranker-8B"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

    def predict(
        self,
        instruction: str = Input(
            description="Task instruction for the reranker (recommended for better performance)",
            default="Given a query and a document, evaluate the relevance of the document to the query."
        ),
        query: str = Input(description="Query text for reranking"),
        documents: str = Input(
            description="JSON string containing list of documents to rerank",
            default='["Document 1 text", "Document 2 text", "Document 3 text"]'
        ),
        top_k: int = Input(
            description="Number of top documents to return",
            ge=1,
            le=100,
            default=5
        ),
        batch_size: int = Input(
            description="Batch size for processing documents",
            ge=1,
            le=32,
            default=8
        ),
    ) -> str:
        """Run text reranking on the model using instruction, query, document format"""
        try:
            # Parse documents from JSON string
            docs_list = json.loads(documents)
            if not isinstance(docs_list, list):
                docs_list = [docs_list]
        except json.JSONDecodeError:
            docs_list = [documents]

        if not docs_list:
            return json.dumps({"error": "No documents provided"})

        # Format each document with instruction, query, doc
        formatted_texts = []
        for doc in docs_list:
            # Use the format from the Hugging Face documentation
            formatted_text = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
            formatted_texts.append(formatted_text)

        rerank_scores = []

        # Process in batches to avoid memory issues
        for i in range(0, len(formatted_texts), batch_size):
            batch_texts = formatted_texts[i:i + batch_size]

            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.model.device)

                outputs = self.model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().float().tolist()

                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]

                rerank_scores.extend(batch_scores)

        # Combine documents with their scores
        scored_docs = []
        for doc, score in zip(docs_list, rerank_scores):
            scored_docs.append({
                "document": doc,
                "score": float(score)
            })

        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        # Return top_k results
        result = {
            "instruction": instruction,
            "query": query,
            "results": scored_docs[:top_k],
            "total_documents": len(docs_list)
        }

        return json.dumps(result, ensure_ascii=False, indent=2)
