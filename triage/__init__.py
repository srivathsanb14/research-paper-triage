"""Personalized Research Paper Triage System.

Pipeline (mirrors the project workflow figure):

    user context ──► interest profile (CONFIGURED)
    paper API    ──► preprocessing ──► paper vectors
    profile + vectors + feedback ──► relevance model (TRAINED)
    scores ──► ranking + triage READ / SKIM / SKIP (TUNED)
    paper + interests + label ──► one-line explanation (CONFIGURED)
    predictions vs. hand labels ──► evaluation (EVALUATED)
    UI feedback ──► labels + feedback log ──► future ranking
"""

__version__ = "1.0.0"
