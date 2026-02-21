from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class TradeTheEventNewsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TradeTheEventNewsRetrieval",
        description="A dataset comprising finance news articles paired with their corresponding headlines and stock ticker symbols.",
        reference="https://aclanthology.org/2021.findings-acl.186.pdf",
        dataset={
            "path": "FinanceMTEB/TradeTheEventNews",
            "revision": "9d499d833355b8ef1810073e5f2aa174d743d4d0",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2021-12-31"),
        domains=["News", "Written", "Financial"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt={"query": "Given a financial news headline, retrieve the full article"},
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
