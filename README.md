# Matryoshka vs. Quantization: A Vector Search Analysis

This project contains the code and experiments for the Towards Data Science article, "Scaling Vector Search: Comparing Quantization and Matryoshka Embeddings for 80% Cost Reduction".

**Read the full article here:** https://towardsdatascience.com/optimizing-vector-search-why-you-should-flatten-structured-data/

## Overview

This project provides a reusable experiment to measure the trade-offs between vector database storage size and retrieval performance. Using techniques like Matryoshka Representation Learning (MRL) and quantization, you can determine the optimal balance of cost and accuracy for your specific use case.

The core logic is contained in the Jupyter Notebook: `notebooks/matryoshka_and_quantization_analysis.ipynb`.

## How the Experiment Works

The notebook follows a clear, automated pipeline:

1.  **Load Data**: It fetches a corpus, queries, and a relevance map (qrels) from a specified Hugging Face dataset.
2.  **Iterate Dimensions**: It loops through a list of embedding dimensions (e.g., `[384, 256, 128, 64]`) provided by the Matryoshka model.
3.  **Build Indexes**: For each dimension, it builds and populates three different types of FAISS indexes:
    *   **No Quantization (Float32)**: The baseline, high-precision index.
    *   **Scalar Quantization (int8)**: An index that approximates vectors using 8-bit integers.
    *   **Binary Quantization (1-bit)**: A highly compressed index using only a single bit per vector component.
4.  **Evaluate**: It measures two key aspects for each index:
    *   **Storage Size**: The on-disk footprint in megabytes.
    *   **Retrieval Performance**: `Recall@10` and `MRR@10` are calculated to measure accuracy.
5.  **Visualize**: The final results are compiled into a pandas DataFrame and used to generate plots that visualize the size-vs-performance trade-offs.

## Running the Experiment

### Default Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/otereshin/matryoshka-quantization-analysis.git
    cd matryoshka-quantization-analysis
    ```

2.  **Install dependencies:**
    The notebook handles its own dependencies in the first cell.

3.  **Launch Jupyter and run the notebook:**
    ```bash
    jupyter notebook
    ```
    Open `notebooks/matryoshka_and_quantization_analysis.ipynb` and execute the cells from top to bottom.

### Running on Your Own Data

The notebook is designed for easy reuse. You can test these techniques on your own data with minimal changes.

#### Option 1: Using a Hugging Face Dataset

If your dataset is on Hugging Face and has a similar structure (`_id`, `text` columns), you can run the experiment by simply changing the constants in the **Configuration** cell of the notebook:

```python
# --- Configuration ---
MODEL_NAME = "mixedbread-ai/mxbai-embed-xsmall-v1"
DATASET_NAME = "your-org/your-dataset-name" # <-- CHANGE THIS
CORPUS_SPLIT = "train"                       # <-- CHANGE THIS
QUERIES_SPLIT = "test"                      # <-- CHANGE THIS
TEXT_COLUMN_NAME = "your_text_column"        # <-- CHANGE THIS
```

#### Option 2: Using Local Files (e.g., CSV)

If you have local files, you just need to load them and adapt them to the structure the notebook expects. The key is to create the `corpus_ds`, `queries_ds`, and `relevant_map` objects.

For example, if you have a local CSV for your corpus, you can replace the `load_and_prepare_data` function with your own loading logic:

```python
# Example of custom data loading function
import pandas as pd
from datasets import Dataset

def load_my_custom_data():
    # 1. Load your corpus from a local file
    # Ensure it has integer '_id' and 'text' columns
    my_corpus_df = pd.read_csv("path/to/my_corpus.csv")
    corpus_ds = Dataset.from_pandas(my_corpus_df)

    # 2. Load your queries
    my_queries_df = pd.read_csv("path/to/my_queries.csv")
    queries_ds = Dataset.from_pandas(my_queries_df)

    # 3. Load your relevance data (qrels)
    # This file should map a query-id to a corpus-id
    qrels_df = pd.read_csv("path/to/my_qrels.csv")
    
    # 4. Build the relevant_map
    indexed_corpus_ids = set(corpus_ds["_id"])
    relevant_map = {}
    for _, row in qrels_df.iterrows():
        q_id, c_id = row["query-id"], row["corpus-id"]
        if c_id in indexed_corpus_ids:
            relevant_map.setdefault(q_id, []).append(c_id)
            
    # 5. Filter queries
    valid_query_ids = list(relevant_map.keys())
    eval_queries = [q for q in queries_ds if q["_id"] in valid_query_ids]

    return corpus_ds, eval_queries, relevant_map

# --- In the notebook, replace the call:
# corpus_ds, eval_queries, relevant_map = load_and_prepare_data()
# --- With your new function:
# corpus_ds, eval_queries, relevant_map = load_my_custom_data()
```
Once your data is loaded into these variables, the rest of the notebook will run without any changes.

## Results

The notebook generates plots that visualize the trade-offs between storage size, retrieval accuracy, and embedding dimensions. These results are discussed in detail in the accompanying article.


## References & Acknowledgments

This experiment is built on the shoulders of several excellent open-source projects, models, and research papers:

* **Embedding Model:** [`mixedbread-ai/mxbai-embed-xsmall-v1`](https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1) - A highly efficient, Matryoshka-enabled embedding model.
* **Dataset:** [`mteb/hotpotqa`](https://huggingface.co/datasets/mteb/hotpotqa) - Used via the Massive Text Embedding Benchmark (MTEB) for evaluating retrieval performance.
* **Vector Search:** [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss) - The core library used for in-memory vector storage, highly efficient similarity search, and vector quantization.
* **Core Concept (MRL):** [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) (Kusupati et al., 2022) - The foundational research paper introducing flexible-dimension embeddings.