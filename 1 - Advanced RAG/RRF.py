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
    """
    rrf_scores.items() -> turns the dictionary into an iterable of (key,value) tupes
    rrf_scores = {'doc_1':2.5, 'doc_2': 4.0, 'doc_3': 1.7} -> [('doc_1',2.5), ('doc_2',4.0), ('doc_3',1.7)]
    key = lambda item: item[1] -> tells sorted to look at the second element of each tuple when sorting
    reverse = True -> sorts in DESC order; reverse = false is default
    """

    return [doc_id for doc_id, score in sorted_docs]

keyword_results = ['docA', 'docB', 'docC'] #results from retriever BM25
semantic_results = ['doc_A', 'doc_B', 'doc_C']#results from retriever SentenceTransformer

ensemble_results = reciprocal_rank_fusion([keyword_results,semantic_results])