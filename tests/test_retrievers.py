from rageval.data import Document
from rageval.retrievers import BM25, tokenize

DOCS = [
    Document("d1", "Vitamin D", "Vitamin D supplementation reduces fracture risk in older adults."),
    Document("d2", "Statins", "Statins lower LDL cholesterol and cardiovascular events."),
    Document("d3", "Sleep", "Short sleep duration is associated with obesity in children."),
    Document("d4", "Vitamin C", "Vitamin C does not prevent the common cold."),
]


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Effect of Vitamin-D on BONE") == ["effect", "vitamin", "d", "bone"]


def test_bm25_ranks_the_matching_document_first():
    hits = BM25(DOCS).search("does vitamin d reduce fractures", k=4)
    assert hits[0][0] == "d1"


def test_bm25_rare_terms_outweigh_common_ones():
    # "vitamin" appears in two docs, "cholesterol" in one: the rare term should win.
    hits = BM25(DOCS).search("vitamin cholesterol", k=4)
    assert hits[0][0] == "d2"


def test_bm25_returns_no_hits_without_term_overlap():
    assert BM25(DOCS).search("quantum chromodynamics", k=4) == []
