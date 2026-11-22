1. Why Header-Aware Chunking (300–500 Tokens) Instead of Fixed Character Splitting?

we split by sections so that logical clauses remain intact. We cap them on 500 tokens tro make sure that no extra noise  dilutes the embedding vector

2. Why Hard Metadata Pre-Filtering Before Search?

We do pre filtering to narrow down the search space before perfdorming vector search. this increases the efficiency of the syustem as it prevents us from searchiung through unrelated vecotrs

3. Why Hybrid Search (Dense + BM25) Instead of Just Vector Search?

Dense embeddings lose precison on exact stirngd (codes, numbers etc). Pairing it with a keyword matching algorithm makes sure we tale into amount both conceoptual questions and exact code lookups.

4. Why Reciprocal Rank Fusion (RRF) Instead of Averaging Raw Scores?

cosine siomilarity outoputs numnbers between 0 and 1, while bm25 outpouts rsandom positive numbers. we cannot cleanly add or average them without complex normalization. RRF ignores the raw scopres completelky and looks only at rank oposition.

5. Why a Cross-Encoder Reranker Instead of Trusting RRF Top-K?

the bi encoders used in vecotr search compleress the query ionto one vector and the documnet into another vector. They never read then two vecotrs together. 

A cross encoder takes the query + chunk into a single transformer pass and performs full tioken by token cross attention.

It is slow , thats is why we run it on the surviving 15 from RRF.

6. Why a Hard Score Cutoff + LLM Bypass Circuit Breaker?

LLMs are trtained to be helpful. They have to give answers tio anything. 

Thats why we put a threshold and if no chuink passess through the threshold, we do not invikle the llm . in this way we dont  give llm a chance to hallucionate and we refuse deterministically.

7. Embedding Provider and Model

The implementation uses OpenRouter's `nvidia/nemotron-3.5-embed-1b:free` for both document and query embeddings, returning 2048-dimensional vectors. This is an explicit implementation choice that supersedes the embedding model named in the technical specification. Re-embedding and a compatible Qdrant collection are required if the embedding model changes.
