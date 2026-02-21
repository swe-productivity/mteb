from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FinNSPClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinNSPClassification",
        description="A Chinese financial negative news and subject determination dataset.",
        reference="https://github.com/alipay/financial_evaluation_dataset/",
        dataset={
            "path": "FinanceMTEB/FinNSP",
            "revision": "1d3ae2b90b692ca702a76f26b94c7cb09b23ca14",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="accuracy",
        date=("2020-01-01", "2023-12-31"),
        domains=["News", "Written", "Financial"],
        task_subtypes=["Topic classification"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Classify a Chinese financial news text as negative news or not",
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
        if "sentence" in self.dataset.column_names.get("test", []):
            self.dataset = self.dataset.rename_column("sentence", "text")
