from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class USNewsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="USNewsRetrieval",
        description="A dataset comprising US financial news articles paired with their corresponding headlines.",
        reference="https://www.kaggle.com/datasets/jeet2016/us-financial-news-articles",
        dataset={
            "path": "FinanceMTEB/USnews",
            "revision": "dda970333494c509262e91d2b44d43430e985b3c",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["News", "Written", "Financial"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a financial news headline, retrieve the full news article"
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
