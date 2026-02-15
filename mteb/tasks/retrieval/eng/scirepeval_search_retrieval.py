from __future__ import annotations

from collections import defaultdict

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
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


class SciRepEvalSearchRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SciRepEvalSearchRetrieval",
        description="Retrieval task using Semantic Scholar search queries to find relevant scientific papers, from the SciRepEval benchmark.",
        reference="https://aclanthology.org/2023.emnlp-main.338/",
        dataset={
            "path": "allenai/scirepeval",
            "revision": "781d35d1bf87253b3dcd0fadcb82bfbee9c244f1",
            "name": "search",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2023-12-31"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=SCIREPEVAL_CITATION,
        prompt={
            "query": "Given a scientific query, identify relevant scientific documents"
        },
    )

    def load_data(self, num_proc: int = 1, **kwargs) -> None:
        if self.data_loaded:
            return

        ds = load_dataset(
            self.metadata.dataset["path"],
            name=self.metadata.dataset["name"],
            revision=self.metadata.dataset["revision"],
        )

        self.dataset = defaultdict(lambda: defaultdict(dict))
        self.dataset["default"]["test"] = self._convert_search_to_retrieval(
            ds["evaluation"]
        )
        self.data_loaded = True

    @staticmethod
    def _convert_search_to_retrieval(
        dataset_split,
    ) -> RetrievalSplitData:
        """Convert SciRepEval search format to MTEB retrieval format."""
        corpus_dict: dict[str, dict[str, str]] = {}
        queries_list: list[dict[str, str]] = []
        relevant_docs: dict[str, dict[str, int]] = {}

        for row in dataset_split:
            query_text = row["query"]
            query_id = str(row["doc_id"])

            queries_list.append({"id": query_id, "text": query_text})
            relevant_docs[query_id] = {}

            for candidate in row["candidates"]:
                cand_id = str(candidate["doc_id"])
                title = candidate.get("title", "") or ""
                abstract = candidate.get("abstract", "") or ""
                corpus_dict[cand_id] = {"title": title, "text": abstract}
                relevant_docs[query_id][cand_id] = int(
                    candidate.get("score", 0)
                )

        corpus = Dataset.from_list(
            [
                {"id": k, "title": v["title"], "text": v["text"]}
                for k, v in corpus_dict.items()
            ]
        )
        queries = Dataset.from_list(queries_list)

        return RetrievalSplitData(
            corpus=corpus,
            queries=queries,
            relevant_docs=relevant_docs,
            top_ranked=None,
        )
