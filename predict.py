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

        # Load model first, then tokenizer (order matters)
        print("Loading AutoModelForSequenceClassification first...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

        # Now load tokenizer and set padding token
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True  # Use fast tokenizer
        )

        # Get tokenizer info and fix padding token
        vocab_size = self.tokenizer.vocab_size
        eos_token_id = self.tokenizer.eos_token_id

        # Try to find a good padding token
        # Common tokens that work well for padding
        preferred_tokens = [
            self.tokenizer.encode(" ", add_special_tokens=False)[0],  # space
            self.tokenizer.encode("\n", add_special_tokens=False)[0],  # newline
            self.tokenizer.encode("<pad>", add_special_tokens=False)[0] if "<pad>" in self.tokenizer.get_vocab() else None,
        ]

        # Find a valid token from our preferred list
        valid_token_id = None
        for token_id in preferred_tokens:
            if token_id is not None and 0 <= token_id < vocab_size:
                valid_token_id = token_id
                valid_token = self.tokenizer.decode([token_id])
                print(f"Using padding token: '{valid_token}' (ID: {valid_token_id})")
                break

        # Fallback to EOS token if it's valid, or use a simple token
        if valid_token_id is None:
            if eos_token_id < vocab_size:
                valid_token_id = eos_token_id
                valid_token = self.tokenizer.eos_token
                print(f"Using EOS token for padding: {valid_token} (ID: {valid_token_id})")
            else:
                # Use a simple safe token - space should always work
                valid_token_id = self.tokenizer.encode(" ", add_special_tokens=False)[0]
                valid_token = " "
                print(f"Using space token for padding (ID: {valid_token_id})")

        # Set padding token explicitly in both tokenizer and model config
        self.tokenizer.pad_token = valid_token
        self.tokenizer.pad_token_id = valid_token_id
        self.model.config.pad_token_id = valid_token_id

        print("✓ Model loaded successfully!")

    def predict(
        self,
        instruction: str = Input(
            description="Task instruction for the reranker (recommended for better performance)",
            default="Given a web search query, retrieve relevant passages that answer the query"
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
        """Run text reranking on the model using standard classification approach"""
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
        pairs = []
        for doc in docs_list:
            formatted_pair = f"{instruction}\nQuery: {query}\nDocument: {doc}"
            pairs.append(formatted_pair)

        rerank_scores = []

        # Process in batches to avoid memory issues
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]

            # Standard classification tokenizer approach
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move to device
            for key in inputs:
                inputs[key] = inputs[key].to(self.model.device)

            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                # For reranking, extract the relevance score
                # For binary classification, we typically want the positive class score
                if len(logits.shape) == 3:  # [batch, seq_len, num_classes]
                    scores = logits[:, -1, 1]  # Last token, positive class
                elif len(logits.shape) == 2:  # [batch, num_classes]
                    scores = logits[:, 1]  # Positive class
                else:
                    scores = logits  # Fallback

                batch_scores = scores.cpu().float().tolist()

                # Ensure we always return a list
                if isinstance(batch_scores, torch.Tensor):
                    batch_scores = batch_scores.tolist()
                elif isinstance(batch_scores, float):
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