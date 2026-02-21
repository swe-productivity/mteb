from mteb.benchmarks.benchmark import Benchmark
from mteb.get_tasks import get_tasks

_FINMTEB_CITATION = r"""
@inproceedings{tang-yang-2025-finmteb,
  author = {Tang, Yixuan and Yang, Yi},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  title = {FinMTEB: Finance Massive Text Embedding Benchmark},
  url = {https://aclanthology.org/2025.emnlp-main.179/},
  year = {2025},
}
"""

FinMTEB = Benchmark(
    name="FinMTEB",
    tasks=get_tasks(
        tasks=[
            # Classification - English
            "ESGClassification",
            "FinancialFraudClassification",
            "FinSentClassification",
            "FiQAABSAClassification",
            "FLSClassification",
            "FOMCClassification",
            "SemEval2017ClassificationFinance",
            # Classification - Chinese
            "FinChinaSentimentClassification",
            "FinFEClassification",
            "FinNSPClassification",
            "OpenFinDataSentimentClassification",
            "Weibo21Classification",
            # Retrieval - English
            "Apple10KRetrieval",
            "HC3Retrieval",
            "TATQARetrieval",
            "TheGoldmanEnRetrieval",
            "TradeTheEventEncyclopediaRetrieval",
            "TradeTheEventNewsRetrieval",
            "USNewsRetrieval",
            # Retrieval - Chinese
            "AlphaFinRetrieval",
            "DISCFinLLMComputingRetrieval",
            "DISCFinLLMRetrieval",
            "DuEEFinRetrieval",
            "FinEvaEncyclopediaRetrieval",
            "FinEvaRetrieval",
            "FinTruthQARetrieval",
            "SmoothNLPRetrieval",
            "TheGoldmanZhRetrieval",
            "THUCNewsRetrieval",
            # Clustering - English
            "ComplaintsClustering",
            "FinanceArxivP2PClustering",
            "FinanceArxivS2SClustering",
            "MInDS14EnClustering",
            "PiiClustering",
            "WikiCompany2IndustryClustering",
            # Clustering - Chinese
            "CCKS2019Clustering",
            "CCKS2020Clustering",
            "CCKS2022Clustering",
            "FinNLClustering",
            "MInDS14ZhClustering",
            # Reranking - English
            "FinFactReranking",
            "FiQA2018Reranking",
            "HC3Reranking",
            # Reranking - Chinese
            "DISCFinLLMReranking",
            "FinEvaReranking",
            # STS - English
            "FINALSts",
            "FinSTSSts",
            # STS - Chinese
            "BQCorpusSts",
            # Summarization - English
            "EctSumSummarization",
            "FINDSumSummarization",
            "FNS2022Summarization",
            # Summarization - Chinese
            "FinEvaHeadlineSummarization",
            "FinEvaSummarization",
            "FiNNASummarization",
            # PairClassification - English
            "HeadlineACPairClassification",
            "HeadlinePDDPairClassification",
            "HeadlinePDUPairClassification",
            # PairClassification - Chinese
            "AFQMCPairClassification",
        ]
    ),
    description="Finance Massive Text Embedding Benchmark (FinMTEB): a comprehensive embedding benchmark for the financial domain covering 7 task types across English and Chinese.",
    reference="https://aclanthology.org/2025.emnlp-main.179/",
    citation=_FINMTEB_CITATION,
    contacts=["yixuantt"],
)
