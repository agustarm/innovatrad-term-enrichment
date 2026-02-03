from __future__ import annotations
from typing import List, Tuple, Iterable, Set
import re
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import AppConfig
from .model_registry import LoadedModel

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\._/]*")
_TFIDF_TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b"

def _is_valid_tfidf_term(term: str) -> bool:
    if len(term) < 3:
        return False
    if not re.search(r"[A-Za-z]", term):
        return False
    if re.fullmatch(r"\d+", term):
        return False
    return True

def simple_tokenize(text: str) -> List[str]:
    # tokenización simple y rápida para candidatos n-gram
    return _WORD_RE.findall(text.lower())

def extract_ngrams(tokens: List[str], nmax: int) -> List[str]:
    out = []
    n = len(tokens)
    for k in range(1, nmax + 1):
        for i in range(0, n - k + 1):
            out.append(" ".join(tokens[i:i+k]))
    return out

def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def score_candidates_binary_classifier(
    mdl: LoadedModel,
    cfg: AppConfig,
    text: str,
    candidates: List[str],
) -> List[Tuple[str, float]]:
    """
    Devuelve [(candidate, prob_keyword)] ordenado por prob desc.
    Entrada al modelo: "candidate [SEP] text"
    """
    tok = mdl.tokenizer
    model = mdl.model
    device = mdl.device

    sep = tok.sep_token or "[SEP]"
    max_len = cfg.finetune.max_length

    batch_size = 32
    scored: List[Tuple[str, float]] = []

    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start:start+batch_size]
            inputs = [f"{c} {sep} {text}" for c in chunk]
            enc = tok(
                inputs,
                truncation=True,
                padding=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits  # [B, num_labels]
            # asumimos label 1 = keyword
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy().tolist()
            scored.extend(list(zip(chunk, probs)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

def finetune_extract_keywords(
    mdl: LoadedModel,
    cfg: AppConfig,
    text: str,
    k: int,
) -> List[str]:
    if not text:
        return []

    tokens = simple_tokenize(text)
    cands = extract_ngrams(tokens, cfg.finetune.ngram_max)

    # filtros rápidos
    cands = [c for c in cands if len(c) >= cfg.finetune.min_token_len]
    cands = dedup_keep_order(cands)

    # cap por doc
    cands = cands[: cfg.finetune.max_candidates_per_doc]
    if not cands:
        return []

    scored = score_candidates_binary_classifier(mdl, cfg, text, cands)

    # threshold + topK
    thr = cfg.finetune.score_threshold
    picked = [c for c, p in scored if p >= thr]
    if len(picked) < k:
        top_all = [c for c, _p in scored]
        merged = dedup_keep_order(picked + top_all)
        return merged[:k]
    return picked[:k]

def baseline_extract_keywords(
    cfg: AppConfig,
    text: str,
    k: int,
) -> List[str]:
    """TF-IDF por documento (ngram 1..3), rápido para demo."""
    if not text:
        return []

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 3),
        max_features=5000,
        token_pattern=_TFIDF_TOKEN_PATTERN,
        stop_words="english",
    )

    X = vectorizer.fit_transform([text])
    if X.nnz == 0:
        tokens = simple_tokenize(text)
        tokens = [t for t in tokens if _is_valid_tfidf_term(t)]
        tokens = dedup_keep_order(tokens)
        return tokens[:k]

    vocab = vectorizer.get_feature_names_out()
    row = X.getrow(0)
    pairs = sorted(zip(row.indices, row.data), key=lambda x: x[1], reverse=True)

    kws: List[str] = []
    for idx, _score in pairs:
        term = vocab[idx]
        if not _is_valid_tfidf_term(term):
            continue
        kws.append(term)
        if len(kws) >= k:
            break
    return kws
