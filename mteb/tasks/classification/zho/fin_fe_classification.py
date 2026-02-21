from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FinFEClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinFEClassification",
        description="A financial social media text sentiment categorization dataset in Chinese.",
        reference="https://arxiv.org/abs/2302.09432",
        dataset={
            "path": "FinanceMTEB/FinFE",
            "revision": "01034e2afdce0f7fa9a51a03aa0fdc1e3d576b05",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="accuracy",
        date=("2020-01-01", "2023-12-31"),
        domains=["Social", "Written", "Financial"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Classify a Chinese financial social media text by its sentiment",
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
