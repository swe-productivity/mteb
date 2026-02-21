from __future__ import annotations

from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

_FINMTEB_CITATION = r"""
@inproceedings{tang-yang-2025-finmteb,
  author = {Tang, Yixuan and Yang, Yi},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  title = {FinMTEB: Finance Massive Text Embedding Benchmark},
  url = {https://aclanthology.org/2025.emnlp-main.179/},
  year = {2025},
}
"""


class HeadlineACPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="HeadlineACPairClassification",
        description="A financial text pair classification dataset for detecting agreement or contradiction between financial headlines.",
        reference="https://huggingface.co/datasets/FinanceMTEB/HeadlinePDU-PairClassification",
        dataset={
            "path": "FinanceMTEB/HeadlinePDU-PairClassification",
            "revision": "afa5a612a01b9c0a8058fba44a3bdf66173583eb",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2020-01-01", "2023-12-31"),
        domains=["News", "Written", "Financial"],
        task_subtypes=["Textual Entailment"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Classify a pair of financial headlines as related or unrelated",
        bibtex_citation=_FINMTEB_CITATION,
    )

    def dataset_transform(self, num_proc: int = 1):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")


class HeadlinePDDPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="HeadlinePDDPairClassification",
        description="A financial text pair classification dataset for detecting price direction discrepancy between financial headlines.",
        reference="https://huggingface.co/datasets/FinanceMTEB/HeadlinePDD-PairClassification",
        dataset={
            "path": "FinanceMTEB/HeadlinePDD-PairClassification",
            "revision": "ad0150ed63940e88846659e46be5138aa0db6c85",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2020-01-01", "2023-12-31"),
        domains=["News", "Written", "Financial"],
        task_subtypes=["Duplicate Detection"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Classify a pair of financial headlines as having the same or different price direction",
        bibtex_citation=_FINMTEB_CITATION,
    )

    def dataset_transform(self, num_proc: int = 1):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")


class HeadlinePDUPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="HeadlinePDUPairClassification",
        description="A financial text pair classification dataset for detecting price direction uncertainty between financial headlines.",
        reference="https://huggingface.co/datasets/FinanceMTEB/HeadlinePDU-PairClassification",
        dataset={
            "path": "FinanceMTEB/HeadlinePDU-PairClassification",
            "revision": "afa5a612a01b9c0a8058fba44a3bdf66173583eb",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2020-01-01", "2023-12-31"),
        domains=["News", "Written", "Financial"],
        task_subtypes=["Duplicate Detection"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Classify a pair of financial headlines as having the same or different price direction uncertainty",
        bibtex_citation=_FINMTEB_CITATION,
    )

    def dataset_transform(self, num_proc: int = 1):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")
