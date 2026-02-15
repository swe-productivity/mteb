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


def _make_text(title: str | None, abstract: str | None) -> str:
    parts = []
    if title:
        parts.append(title)
    if abstract:
        parts.append(abstract)
    return ". ".join(parts)


def _convert_ranking_to_retrieval(
    dataset_split,
) -> RetrievalSplitData:
    """Convert SciRepEval ranking format (query + candidates) to MTEB retrieval format."""
    corpus_dict: dict[str, dict[str, str]] = {}
    queries_list: list[dict[str, str]] = []
    relevant_docs: dict[str, dict[str, int]] = {}
    top_ranked: dict[str, list[str]] = {}

    for row in dataset_split:
        query = row["query"]
        query_id = str(query["doc_id"])
        query_text = _make_text(query.get("title"), query.get("abstract"))

        # Add query to corpus and queries
        corpus_dict[query_id] = {"title": query.get("title", "") or "", "text": query.get("abstract", "") or ""}
        queries_list.append({"id": query_id, "text": query_text})

        # Process candidates
        relevant_docs[query_id] = {}
        top_ranked[query_id] = []
        for candidate in row["candidates"]:
            cand_id = str(candidate["doc_id"])
            corpus_dict[cand_id] = {"title": candidate.get("title", ""), "text": candidate.get("abstract", "") or ""}
            relevant_docs[query_id][cand_id] = int(candidate.get("score", 0))
            top_ranked[query_id].append(cand_id)

    corpus = Dataset.from_list(
        [{"id": k, "title": v["title"], "text": v["text"]} for k, v in corpus_dict.items()]
    )
    queries = Dataset.from_list(queries_list)

    return RetrievalSplitData(
        corpus=corpus,
        queries=queries,
        relevant_docs=relevant_docs,
        top_ranked=top_ranked,
    )


class SciRepEvalHighInfluenceCiteReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SciRepEvalHighInfluenceCiteReranking",
        description="Reranking task to identify highly influential citations among candidate papers for a given query paper, from the SciRepEval benchmark.",
        reference="https://aclanthology.org/2023.emnlp-main.338/",
        dataset={
            "path": "allenai/scirepeval",
            "revision": "781d35d1bf87253b3dcd0fadcb82bfbee9c244f1",
            "name": "high_influence_cite",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_1000",
        date=("2020-01-01", "2023-12-31"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=["Scientific Reranking"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=SCIREPEVAL_CITATION,
        prompt={
            "query": "Given a scientific paper title and abstract, retrieve papers that are highly influential citations in the given paper"
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
        self.dataset["default"]["test"] = _convert_ranking_to_retrieval(
            ds["evaluation"]
        )
        self.data_loaded = True
