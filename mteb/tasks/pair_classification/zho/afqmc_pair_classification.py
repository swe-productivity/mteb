from __future__ import annotations

from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AFQMCPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="AFQMCPairClassification",
        description="Ant Financial Question Matching Corpus: a Chinese pair classification dataset for financial question matching.",
        reference="https://tianchi.aliyun.com/dataset/106411",
        dataset={
            "path": "FinanceMTEB/AFQMC-PairClassification",
            "revision": "623887e33b741cf9e5faa2bae12a4269c1de8fec",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="max_ap",
        date=("2018-01-01", "2021-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Duplicate Detection"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Classify a pair of Chinese financial questions as semantically equivalent or not",
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

    def dataset_transform(self, num_proc: int = 1):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")
