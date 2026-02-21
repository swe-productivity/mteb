from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class Weibo21Classification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Weibo21Classification",
        description="A Chinese fake news detection dataset in the finance domain from Weibo.",
        reference="https://dl.acm.org/doi/pdf/10.1145/3459637.3482139",
        dataset={
            "path": "FinanceMTEB/MDFEND-Weibo21",
            "revision": "db799d3d74bc752cb30b264a6254ab52471f693d",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="accuracy",
        date=("2021-01-01", "2021-12-31"),
        domains=["Social", "Written", "Financial"],
        task_subtypes=["Topic classification"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Classify a Chinese financial social media post as fake news or real news",
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
