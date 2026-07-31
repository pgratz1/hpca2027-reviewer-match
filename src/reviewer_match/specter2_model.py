"""SPECTER2 loading and batched text encoding.

No reviewer/paper knowledge here — just "list of strings in, (N, 768) array
out". Callers are responsible for shaping their inputs into the model's
trained format: ``title + tokenizer.sep_token + abstract`` (empty abstract
is fine when none is available, e.g. a bare DBLP title).

Uses the `allenai/specter2_base` encoder with the `allenai/specter2`
"proximity" adapter (document-to-document similarity — the one named plainly
`specter2` on the Hub, as opposed to `_adhoc_query`, `_classification`, or
`_regression`), loaded via the `adapters` (AdapterHub) library.
"""

from __future__ import annotations

import numpy as np
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer, PreTrainedTokenizerBase

BASE_MODEL = "allenai/specter2_base"
PROXIMITY_ADAPTER = "allenai/specter2"


def load_model(device: str = "cuda") -> tuple[PreTrainedTokenizerBase, torch.nn.Module]:
    """Load the SPECTER2 base encoder with the proximity adapter active."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoAdapterModel.from_pretrained(BASE_MODEL)
    model.load_adapter(PROXIMITY_ADAPTER, source="hf", set_active=True)
    model = model.to(device)
    model.eval()
    return tokenizer, model


def encode_texts(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    batch_size: int = 16,
    max_length: int = 512,
) -> np.ndarray:
    """Encode `texts` into an (N, 768) array of CLS-pooled SPECTER2 embeddings."""
    device = next(model.parameters()).device
    chunks: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
        cls_embeddings = out.last_hidden_state[:, 0, :]
        chunks.append(cls_embeddings.cpu().numpy())

    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 768), dtype=np.float32)
