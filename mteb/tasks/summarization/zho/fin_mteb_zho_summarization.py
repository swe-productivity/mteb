from __future__ import annotations

from mteb.abstasks.sts import AbsTaskSTS
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


class FinEvaHeadlineSummarization(AbsTaskSTS):
    min_score = 0
    max_score = 1

    metadata = TaskMetadata(
        name="FinEvaHeadlineSummarization",
        description="A Chinese financial news headline summarization dataset.",
        reference="https://github.com/alipay/financial_evaluation_dataset/",
        dataset={
            "path": "FinanceMTEB/FinEvaHeadline",
            "revision": "4c72b5793c541449d3289e4d86f54883fa633313",
        },
        type="Summarization",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="cosine_spearman",
        date=("2020-01-01", "2023-12-31"),
        domains=["News", "Written", "Financial"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_FINMTEB_CITATION,
    )

    def dataset_transform(self, num_proc: int = 1, **kwargs):
        self.dataset = self.dataset.rename_columns(
            {"text": "sentence1", "summary": "sentence2"}
        )


class FinEvaSummarization(AbsTaskSTS):
    min_score = 0
    max_score = 1

    metadata = TaskMetadata(
        name="FinEvaSummarization",
        description="A Chinese financial news summarization dataset.",
        reference="https://github.com/alipay/financial_evaluation_dataset/",
        dataset={
            "path": "FinanceMTEB/FinEvaSum",
            "revision": "094d9c825e66d53c17cd3c3bb258523b8a12af15",
        },
        type="Summarization",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="cosine_spearman",
        date=("2020-01-01", "2023-12-31"),
        domains=["News", "Written", "Financial"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_FINMTEB_CITATION,
    )

    def dataset_transform(self, num_proc: int = 1, **kwargs):
        self.dataset = self.dataset.rename_columns(
            {"text": "sentence1", "summary": "sentence2"}
        )


class FiNNASummarization(AbsTaskSTS):
    min_score = 0
    max_score = 1

    metadata = TaskMetadata(
        name="FiNNASummarization",
        description="A Chinese financial news summarization dataset.",
        reference="https://github.com/ssymmetry/BBT-FinCUGE-Applications/blob/main/FinCUGE_Publish/finna/train_list.json",
        dataset={
            "path": "FinanceMTEB/FinNA",
            "revision": "6c7e4c5807dbe5c53f9d4d445a9f707607f7e286",
        },
        type="Summarization",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="cosine_spearman",
        date=("2020-01-01", "2023-12-31"),
        domains=["News", "Written", "Financial"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_FINMTEB_CITATION,
    )

    def dataset_transform(self, num_proc: int = 1, **kwargs):
        self.dataset = self.dataset.rename_columns(
            {"text": "sentence1", "summary": "sentence2"}
        )
