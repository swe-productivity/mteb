from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class Apple10KRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Apple10KRetrieval",
        description="A RAG benchmark for finance applications based on Apple's 10-K annual report.",
        reference="https://arxiv.org/pdf/2301.07597",
        dataset={
            "path": "FinanceMTEB/Apple-10K-2022",
            "revision": "27d0b84029e5de607fc7a8e2fb2a315e9b71f570",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a financial question, retrieve relevant passages from the annual report"
        },
        bibtex_citation=r"""
@inproceedings{tang-yang-2025-finmteb,
  author = {Tang, Yixuan and Yang, Yi},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  title = {FinMTEB: Finance Massive Text Embedding Benchmark},
  url = {https://aclanthology.org/2025.emnlp-main.179/},
  year = {2025},
}
""",
    )
