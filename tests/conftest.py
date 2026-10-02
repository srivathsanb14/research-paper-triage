import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage.models import InterestProfile, Paper  # noqa: E402


def make_paper(i: int, title: str, abstract: str, published: str = "2026-09-01") -> Paper:
    return Paper(id=f"arxiv:2609.{i:05d}", title=title, abstract=abstract, authors=["A. Author"], published=published, source="arxiv")


RAG = [
    ("Retrieval-Augmented Generation for Scientific Question Answering",
     "We present a retrieval-augmented generation pipeline that answers questions over scientific papers using dense retrieval and a reranker."),
    ("Evaluating Retrieval Quality in RAG Pipelines",
     "We study how retrieval quality affects the faithfulness of answers in retrieval augmented generation systems and propose new evaluation metrics."),
    ("Dense Passage Retrieval with Hard Negatives for Open-Domain QA",
     "Dense passage retrieval improves open-domain question answering when trained with hard negatives mined from BM25."),
    ("LLM Agents that Search: Tool Use for Multi-hop Question Answering",
     "Large language model agents call search tools iteratively to answer multi-hop questions and cite retrieved evidence."),
    ("Reranking Retrieved Passages with Cross-Encoders",
     "Cross-encoder rerankers improve precision of retrieved passages for question answering and retrieval augmented generation."),
]
MID = [
    ("Efficient Transformer Inference on Edge Devices",
     "We quantize large language models to run inference on edge devices with small accuracy loss."),
    ("A Survey of Instruction Tuning for Language Models",
     "Instruction tuning aligns language models with human instructions; we survey datasets and methods."),
    ("Knowledge Graph Completion with Language Models",
     "We use language models to complete knowledge graphs by predicting missing links between entities."),
]
OFF = [
    ("Quadrotor Control with Model Predictive Control",
     "We design a model predictive controller for agile quadrotor flight in cluttered outdoor environments."),
    ("Legged Robot Locomotion via Reinforcement Learning",
     "A quadruped robot learns to walk over rough terrain with deep reinforcement learning in simulation."),
    ("Protein Structure Prediction with Diffusion Models",
     "We generate protein backbones with an equivariant diffusion model and evaluate designability."),
    ("Medical Image Segmentation with U-Nets",
     "A U-Net variant segments tumours in MRI scans with improved Dice scores on a hospital dataset."),
    ("Speech Enhancement in Noisy Environments",
     "We denoise speech recordings with a convolutional recurrent network trained on synthetic noise."),
    ("Galaxy Morphology Classification",
     "Convolutional networks classify galaxy images from sky surveys into morphological classes."),
]


@pytest.fixture
def corpus() -> list[Paper]:
    out, i = [], 0
    for group in (RAG, MID, OFF):
        for t, a in group:
            out.append(make_paper(i, t, a))
            i += 1
    return out


@pytest.fixture
def rag_profile() -> InterestProfile:
    return InterestProfile(
        name="rag",
        description="I build retrieval-augmented generation systems for question answering over scientific documents.",
        keywords=["retrieval augmented generation", "question answering", "dense retrieval"],
        focus="evaluating retrieval quality in RAG pipelines",
        avoid=["robot"],
        hours_per_week=2,
    )
