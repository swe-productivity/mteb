from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FinancialFraudClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinancialFraudClassification",
        description="A dataset for detecting financial fraud in text.",
        reference="https://github.com/amitkedia007/Financial-Fraud-Detection-Using-LLMs",
        dataset={
            "path": "FinanceMTEB/FinancialFraud",
            "revision": "e569a69e058ad8504f03556cd05c36700767d193",
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
        prompt="Classify a given financial text as fraudulent or legitimate",
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

    def dataset_transform(self, num_proc: int = 1, **kwargs):
        self.dataset = self.dataset.rename_column("sentence", "text")
