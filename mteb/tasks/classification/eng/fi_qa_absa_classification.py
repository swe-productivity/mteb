from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FiQAABSAClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FiQAABSAClassification",
        description="A polar sentiment dataset of sentences from financial news, categorized by sentiment into positive, negative, or neutral, based on the FiQA aspect-based sentiment analysis benchmark.",
        reference="https://sites.google.com/view/fiqa/home",
        dataset={
            "path": "FinanceMTEB/FiQA_ABSA",
            "revision": "afa907ab4c6441afb8ee70bd99802bb707d3d2ab",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-12-31"),
        domains=["Web", "Written", "Financial"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Classify a financial question or opinion by its sentiment",
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
