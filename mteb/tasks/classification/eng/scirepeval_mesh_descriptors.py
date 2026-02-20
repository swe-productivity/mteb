from __future__ import annotations

from datasets import DatasetDict

from mteb.abstasks.classification import AbsTaskClassification
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


class SciRepEvalMeSHDescriptorsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SciRepEvalMeSHDescriptorsClassification",
        description="Classification of scientific papers by their MeSH (Medical Subject Headings) descriptors, from the SciRepEval benchmark.",
        reference="https://aclanthology.org/2023.emnlp-main.338/",
        dataset={
            "path": "allenai/scirepeval",
            "revision": "781d35d1bf87253b3dcd0fadcb82bfbee9c244f1",
            "name": "mesh_descriptors",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="f1",
        date=("2020-01-01", "2023-12-31"),
        domains=["Academic", "Medical", "Non-fiction", "Written"],
        task_subtypes=["Topic classification"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=SCIREPEVAL_CITATION,
        prompt="Classify a given scientific document",
    )

    def dataset_transform(self, num_proc: int = 1):
        # Build label mapping from all splits
        all_descriptors = set()
        for split_name in self.dataset:
            all_descriptors.update(self.dataset[split_name]["descriptor"])
        descriptor_to_int = {d: i for i, d in enumerate(sorted(all_descriptors))}

        # Combine title and abstract into text, map descriptor to integer label
        self.dataset = self.dataset.map(
            lambda x: {
                "text": (x["title"] or "") + ". " + (x["abstract"] or ""),
                "label": descriptor_to_int[x["descriptor"]],
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
