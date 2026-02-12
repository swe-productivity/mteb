from __future__ import annotations

from datasets import DatasetDict

from mteb.abstasks.multilabel_classification import AbsTaskMultilabelClassification
from mteb.abstasks.task_metadata import TaskMetadata

SCIREPEVAL_CITATION = r"""
@inproceedings{singh-etal-2023-scirepeval,
  author = {Singh, Amanpreet and D'Arcy, Mike and Cohan, Arman and Downey, Doug and Feldman, Sergey},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/2023.emnlp-main.338},
  pages = {5548--5566},
  title = {{SciRepEval}: A Multi-Format Benchmark for Scientific Document Representations},
  year = {2023},
}
"""


class SciRepEvalFoSClassification(AbsTaskMultilabelClassification):
    metadata = TaskMetadata(
        name="SciRepEvalFoSClassification",
        description="Multi-label classification of scientific papers by Field of Study (e.g. Computer Science, Physics, Biology), from the SciRepEval benchmark.",
        reference="https://aclanthology.org/2023.emnlp-main.338/",
        dataset={
            "path": "allenai/scirepeval",
            "revision": "781d35d1bf87253b3dcd0fadcb82bfbee9c244f1",
            "name": "fos",
        },
        type="MultilabelClassification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2023-12-31"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=["Topic classification"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=SCIREPEVAL_CITATION,
        prompt="Classify scientific papers by their field of study based on title and abstract",
    )

    def dataset_transform(self, num_proc: int = 1):
        # Combine title and abstract into text, rename labels to label
        self.dataset = self.dataset.map(
            lambda x: {
                "text": (x["title"] or "")
                + ". "
                + (x["abstract"] or ""),
                "label": x["labels"],
            },
            num_proc=num_proc,
        )
        # Rename evaluation split to test
        self.dataset = DatasetDict(
            {
                "train": self.dataset["train"],
                "test": self.dataset["evaluation"],
            }
        )
        # Subsample for efficiency
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train", "test"]
        )
