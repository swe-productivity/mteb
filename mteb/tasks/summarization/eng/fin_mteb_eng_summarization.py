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


class EctSumSummarization(AbsTaskSTS):
    min_score = 0
    max_score = 1

    metadata = TaskMetadata(
        name="EctSumSummarization",
        description="A dataset for bullet point summarization of long earnings call transcripts.",
        reference="https://arxiv.org/abs/2210.12467",
        dataset={
            "path": "FinanceMTEB/ECTsum",
            "revision": "036a3cbc49ce9e77af1832693a32bfdf6207bb57",
        },
        type="Summarization",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2022-01-01", "2022-12-31"),
        domains=["Written", "Financial"],
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


class FINDSumSummarization(AbsTaskSTS):
    min_score = 0
    max_score = 1

    metadata = TaskMetadata(
        name="FINDSumSummarization",
        description="A large-scale dataset for long text and multi-table financial summarization.",
        reference="https://aclanthology.org/2022.findings-emnlp.145/",
        dataset={
            "path": "FinanceMTEB/FINDsum",
            "revision": "0ca87ef0286fc1451841761a684548dc1ca36070",
        },
        type="Summarization",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2022-01-01", "2022-12-31"),
        domains=["Written", "Financial"],
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


class FNS2022Summarization(AbsTaskSTS):
    min_score = 0
    max_score = 1

    metadata = TaskMetadata(
        name="FNS2022Summarization",
        description="Financial Narrative Summarisation for 10-K annual reports.",
        reference="https://wp.lancs.ac.uk/cfie/fns2022/",
        dataset={
            "path": "FinanceMTEB/FNS",
            "revision": "3a6eaec26efc16f5668d0c1511b4309152998466",
        },
        type="Summarization",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2022-01-01", "2022-12-31"),
        domains=["Written", "Financial"],
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
