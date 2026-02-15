from __future__ import annotations

from datasets import DatasetDict

from mteb.abstasks.regression import AbsTaskRegression
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


class SciRepEvalPeerReviewScoreRegression(AbsTaskRegression):
    metadata = TaskMetadata(
        name="SciRepEvalPeerReviewScoreRegression",
        description="Regression task predicting the mean peer review score of a scientific paper from its title and abstract, from the SciRepEval benchmark.",
        reference="https://aclanthology.org/2023.emnlp-main.338/",
        dataset={
            "path": "allenai/scirepeval",
            "revision": "781d35d1bf87253b3dcd0fadcb82bfbee9c244f1",
            "name": "peer_review_score_hIndex",
        },
        type="Regression",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="kendalltau",
        date=("2020-01-01", "2023-12-31"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=SCIREPEVAL_CITATION,
        prompt="Given a scientific document, predict a metric",
    )

    def dataset_transform(self, num_proc: int = 1):
        self.dataset = DatasetDict(
            {"evaluation": self.dataset["evaluation"]}
        )
        # Combine title and abstract into text, use mean_rating as value
        self.dataset = self.dataset.map(
            lambda x: {
                "text": (x["title"] or "")
                + ". "
                + (x["abstract"] or ""),
                "value": float(x["mean_rating"]),
            },
            num_proc=num_proc,
        )
        # Split evaluation into train and test
        split = self.dataset["evaluation"].train_test_split(
            test_size=0.5, seed=self.seed
        )
        self.dataset = DatasetDict(
            {
                "train": split["train"],
                "test": split["test"],
            }
        )
        # Subsample for efficiency
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train", "test"]
        )
