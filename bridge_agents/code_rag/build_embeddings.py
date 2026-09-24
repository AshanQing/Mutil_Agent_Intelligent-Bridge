from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embedding import DEFAULT_EMBEDDING_MODEL, SentenceTransformerBackend
from .paths import DEFAULT_EVIDENCE_DOCUMENTS_PATH, DEFAULT_INDEX_DIR, DEFAULT_MODEL_CACHE_DIR
from .vector_index import build_vector_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the code RAG semantic vector index.")
    parser.add_argument(
        "--evidence",
        default=str(DEFAULT_EVIDENCE_DOCUMENTS_PATH),
        help="Path to prompt-ready evidence_documents.jsonl.",
    )
    parser.add_argument(
        "--index-dir",
        default=str(DEFAULT_INDEX_DIR),
        help="Directory containing the keyword index manifest.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence Transformers model ID or local model directory.",
    )
    parser.add_argument("--device", default="cpu", help="Embedding device, for example cpu or cuda.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_MODEL_CACHE_DIR),
        help="Writable local model cache directory.",
    )
    args = parser.parse_args()

    result = build_vector_index(
        evidence_path=Path(args.evidence),
        index_dir=Path(args.index_dir),
        backend=SentenceTransformerBackend(
            args.model,
            device=args.device,
            batch_size=args.batch_size,
            cache_dir=Path(args.cache_dir),
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
