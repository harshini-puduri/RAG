# Reciprocal Rank Fusion
def reciprocal_rank_fusion(ranked_lists:list[list], k:int)->list:
    # k=60 in research papers. it's used to down-weight documents with lower ranks
    rrf_scores = {} #to store rrf scores
    for i in ranked_lists:
        for rank, doc_id in enumerate(i, 1):
            score = 1 / (k+rank)
            if doc_id in rrf_scores:
                rrf_scores[doc_id] += score
            else: rrf_scores[doc_id] = score
        
    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    return [doc_id for doc_id, score in sorted_docs]

keyword_results = #results from retriever BM25
semantic_results = #results from rrf

ensemble_results = reciprocal_rank_fusion([keyword_results,semantic_results])