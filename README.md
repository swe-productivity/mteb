<h1 align="center">Massive Text Embedding Benchmark</h1>

<p align="center">
    <a href="https://github.com/embeddings-benchmark/mteb/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/embeddings-benchmark/mteb.svg">
    </a>
    <a href="https://arxiv.org/abs/2210.07316">
        <img alt="GitHub release" src="https://img.shields.io/badge/arXiv-2305.14251-b31b1b.svg">
    </a>
    <a href="https://github.com/embeddings-benchmark/mteb/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/embeddings-benchmark/mteb.svg?color=green">
    </a>
    <a href="https://pepy.tech/project/mteb">
        <img alt="Downloads" src="https://static.pepy.tech/personalized-badge/mteb?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="#installation">Installation</a> |
        <a href="#usage-documentation">Usage</a> |
        <a href="https://huggingface.co/spaces/mteb/leaderboard">Leaderboard</a> |
        <a href="#documentation">Documentation</a> |
        <a href="#citing">Citing</a>
    <p>
</h4>

<h3 align="center">
    <a href="https://huggingface.co/spaces/mteb/leaderboard"><img style="float: middle; padding: 10px 10px 10px 10px;" width="60" height="55" src="./docs/images/hf_logo.png" /></a>
</h3>


## Installation

```bash
pip install mteb
```


## Example Usage


### Using a script

```python
import mteb
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
model_name = "average_word_embeddings_komninos"

model = mteb.get_model(model_name) # if the model is not implemented in MTEB it will be eq. to SentenceTransformer(model_name)
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")
```

### Using the CLI

```bash
mteb available_tasks # list _all_ available tasks

mteb run -m sentence-transformers/all-MiniLM-L6-v2 \
    -t Banking77Classification  \
    --verbosity 3

# if nothing is specified default to saving the results in the results/{model_name} folder
```

Note that using multiple GPUs in parallel can be done by just having a custom encode function that distributes the inputs to multiple GPUs like e.g. [here](https://github.com/microsoft/unilm/blob/b60c741f746877293bb85eed6806736fc8fa0ffd/e5/mteb_eval.py#L60) or [here](https://github.com/ContextualAI/gritlm/blob/09d8630f0c95ac6a456354bcb6f964d7b9b6a609/gritlm/gritlm.py#L75). See [custom models](docs/usage/usage.md#using-a-custom-model) for more information.


## Usage Documentation
The following links to the main sections in the usage documentation.

| Section | |
| ------- |- |
| **General** | |
| [Evaluating a Model](docs/usage/usage.md#evaluating-a-model) | How to evaluate a model |
| [Evaluating on different Modalities](docs/usage/usage.md#evaluating-on-different-modalities) | How to evaluate image and image-text tasks |
| [MIEB](docs/mieb/readme.md) | How to run the Massive Image Embedding Benchmark |
| **Selecting Tasks** | |
| [Selecting a benchmark](docs/usage/usage.md#selecting-a-benchmark) | How to select benchmarks |
| [Task selection](docs/usage/usage.md#task-selection) | How to select and filter tasks |
| [Selecting Split and Subsets](docs/usage/usage.md#selecting-evaluation-split-or-subsets) | How to select evaluation splits or subsets |
| [Using a Custom Task](docs/usage/usage.md#using-a-custom-task) | How to evaluate on a custom task |
| **Selecting a Model** | |
| [Using a Pre-defined Model](docs/usage/usage.md#using-a-pre-defined-model) | How to run a pre-defined model |
| [Using a SentenceTransformer Model](docs/usage/usage.md#using-a-sentence-transformer-model) | How to run a model loaded using sentence-transformers |
| [Using a Custom Model](docs/usage/usage.md#using-a-custom-model) | How to run and implement a custom model |
| **Running Evaluation** | |
| [Passing Arguments to the model](docs/usage/usage.md#passing-in-encode-arguments) | How to pass `encode` arguments to the model |
| [Running Cross Encoders](docs/usage/usage.md#running-cross-encoders-on-reranking) | How to run cross encoders for reranking |
| [Running Late Interaction (ColBERT)](docs/usage/usage.md#using-late-interaction-models) | How to run late interaction models |
| [Saving Retrieval Predictions](docs/usage/usage.md#saving-retrieval-task-predictions) | How to save prediction for later analysis |
| [Caching Embeddings](docs/usage/usage.md#caching-embeddings-to-re-use-them) | How to cache and re-use embeddings |
| **Leaderboard** | |
| [Running the Leaderboard Locally](docs/usage/usage.md#running-the-leaderboard-locally) | How to run the leaderboard locally |
| [Report Data Contamination](docs/usage/usage.md#annotate-contamination) | How to report data contamination for a model |
| [Loading and working with Results](docs/usage/results.md) | How to load and working with the raw results from the leaderboard, including making result dataframes |



## Overview

| Overview                       |                                                                                     |
|--------------------------------|-------------------------------------------------------------------------------------|
| 📈 [Leaderboard]               | The interactive leaderboard of the benchmark                                        |
| 📋 [Tasks]                     | Overview of available tasks                                                         |
| 📐 [Benchmarks]                | Overview of available benchmarks                                                    |
| **Contributing**               |                                                                                     |
| 🤖 [Adding a model]            | How to submit a model to MTEB and to the leaderboard                                |
| 👩‍🔬 [Reproducible workflows]    | How to create reproducible workflows with MTEB                                      |
| 👩‍💻 [Adding a dataset]          | How to add a new task/dataset to MTEB                                               |
| 👩‍💻 [Adding a benchmark]        | How to add a new benchmark to MTEB and to the leaderboard                           |
| 🤝 [Contributing]              | How to contribute to MTEB and set it up for development                             |

[Tasks]: docs/tasks.md
[Benchmarks]: docs/benchmarks.md
[Contributing]: CONTRIBUTING.md
[Adding a model]: docs/adding_a_model.md
[Adding a dataset]: docs/adding_a_dataset.md
[Adding a benchmark]: docs/adding_a_benchmark.md
[Leaderboard]: https://huggingface.co/spaces/mteb/leaderboard
[Reproducible workflows]: docs/reproducible_workflow.md

