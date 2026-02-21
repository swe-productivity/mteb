from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FOMCClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FOMCClassification",
        description="A task for hawkish-dovish classification of Federal Open Market Committee (FOMC) statements in the finance domain.",
        reference="https://github.com/gtfintechlab/fomc-hawkish-dovish",
        dataset={
            "path": "FinanceMTEB/FOMC",
            "revision": "cdaf1306a24bc5e7441c7c871343efdf4c721bc2",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2022-12-31"),
        domains=["Government", "Written", "Financial"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Classify a financial statement as hawkish, dovish, or neutral",
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
