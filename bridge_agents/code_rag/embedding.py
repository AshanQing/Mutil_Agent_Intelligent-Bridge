from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .paths import DEFAULT_MODEL_CACHE_DIR


BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


def resolve_cached_model_path(model_name: str, cache_dir: Path) -> Path:
    direct_path = Path(model_name)
    if direct_path.exists():
        return direct_path
    model_root = Path(cache_dir) / f"models--{model_name.replace('/', '--')}"
    ref_path = model_root / "refs" / "main"
    if not ref_path.exists():
        raise FileNotFoundError(f"Cached model revision does not exist: {ref_path}")
    revision = ref_path.read_text(encoding="utf-8").strip()
    snapshot_path = model_root / "snapshots" / revision
    if not revision or not snapshot_path.exists():
        raise FileNotFoundError(f"Cached model snapshot does not exist: {snapshot_path}")
    return snapshot_path


class SentenceTransformerBackend:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        model: Any | None = None,
        device: str = "cpu",
        batch_size: int = 32,
        query_instruction: str = BGE_QUERY_INSTRUCTION,
        cache_dir: Path = DEFAULT_MODEL_CACHE_DIR,
        offline: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.cache_dir = Path(cache_dir)
        self.offline = offline
        self._model = model

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"{self.query_instruction}{text}" for text in texts])

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        return np.asarray(
            model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def _load_model(self) -> Any:
        if self._model is None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(self.cache_dir / ".hf"))
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(self.cache_dir / ".hf" / "hub"))
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required to build or query the vector index."
                ) from exc
            model_source = (
                str(resolve_cached_model_path(self.model_name, self.cache_dir))
                if self.offline
                else self.model_name
            )
            self._model = SentenceTransformer(
                model_source,
                device=self.device,
                cache_folder=str(self.cache_dir),
            )
        return self._model
