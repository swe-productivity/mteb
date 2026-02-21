from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FLSClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FLSClassification",
        description="A finance dataset for detecting whether a sentence is a forward-looking statement.",
        reference="https://huggingface.co/datasets/FinanceMTEB/FLS",
        dataset={
            "path": "FinanceMTEB/FLS",
            "revision": "39b6719f1d7197df4498fea9fce20d4ad782a083",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2023-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Topic classification"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Classify a financial sentence as forward-looking or not",
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
