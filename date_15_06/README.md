# Hybrid RAG Long-Context Evaluation

This folder contains `Colab2_Hybrid_RAG_LongContext_Claude_v2.py`, a notebook-exported script that evaluates a hybrid RAG pipeline and a full-context baseline.

## What the combined plot shows

The chart `context_eval_tokens_precision_relevance_latency.png` compares five metrics for the first eight evaluation queries:

- `Total_Tokens`
- `Context_Precision`
- `Answer_Relevance`
- `Faithfulness`
- `Latency_ms`

Because these metrics use different units, the plot normalizes them to a 0-1 range so their trends can be compared in one figure.

## How to read it

- `Total_Tokens` shows how much model context was used for each query.
- `Context_Precision` shows how many retrieved chunks were actually relevant.
- `Answer_Relevance` reflects lexical overlap with the reference answer, using BLEU and ROUGE-L.
- `Faithfulness` shows how well the answer is supported by the supplied context.
- `Latency_ms` shows the end-to-end time per query.

## Main insight

The plot is meant to reveal whether better retrieval quality also leads to better answer quality.

- If `Context_Precision` and `Faithfulness` move together, the retriever is grounding answers well.
- If `Total_Tokens` rises sharply while `Context_Precision` stays flat, the prompt may be carrying extra context without improving quality.
- If `Latency_ms` increases with token count, generation cost is likely the dominant bottleneck.

In short, the graph helps answer two practical questions:

1. Is the retriever bringing back useful context?
2. Does that context improve answer quality enough to justify the extra latency and token usage?

