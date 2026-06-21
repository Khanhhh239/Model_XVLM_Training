"""Inference: encode+cache (T4-free workflow), fusion/re-rank, and X-VLM ITM rerank."""
from .encode import (
    average_features,
    encode_image_loader,
    encode_retrieval_images,
    encode_retrieval_text,
    encode_text_list,
    load_embeddings,
    make_distractor_eval_fn,
    save_embeddings,
)
from .pipeline import build_ranking, evaluate_with_pipeline, fuse_and_rerank, write_answer
from .rerank_xvlm import itm_rerank_ranking, itm_scores_topk

__all__ = [
    "encode_image_loader",
    "encode_text_list",
    "encode_retrieval_images",
    "encode_retrieval_text",
    "make_distractor_eval_fn",
    "average_features",
    "save_embeddings",
    "load_embeddings",
    "fuse_and_rerank",
    "build_ranking",
    "evaluate_with_pipeline",
    "write_answer",
    "itm_scores_topk",
    "itm_rerank_ranking",
]
