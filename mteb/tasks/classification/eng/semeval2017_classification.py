from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SemEval2017ClassificationFinance(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SemEval2017ClassificationFinance",
        description="A polar sentiment dataset of financial news headlines, categorized by sentiment into positive, negative, or neutral, from SemEval-2017 Task 5.",
        reference="https://alt.qcri.org/semeval2017/task5/",
        dataset={
            "path": "FinanceMTEB/SemEva2017_Headline",
            "revision": "f0e198ba04c23d949ef803ce32ee1e4f2d8d3696",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2017-01-01", "2017-12-31"),
        domains=["News", "Written", "Financial"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Classify a financial news headline by its sentiment",
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
