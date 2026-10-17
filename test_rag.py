
from rag import load_documents, SafetyKnowledgeBase


def test_loads_seven_guidelines():
    documents = load_documents()
    assert len(documents) == 7


def test_guidelines_have_no_duplicate_numbering():
    documents = load_documents()
    assert documents[0].startswith("Identify visible road users")
    assert documents[3].startswith("Seatbelt compliance")
    assert not documents[3].startswith("4.")


def test_empty_query_returns_no_results():
    knowledge_base = SafetyKnowledgeBase()
    assert knowledge_base.search("") == []
    assert knowledge_base.search("   ") == []


def test_seatbelt_query_retrieves_relevant_guideline():
    knowledge_base = SafetyKnowledgeBase()
    results = knowledge_base.search(
        "What should I check before declaring a seatbelt violation?"
    )
    assert results
    assert "Seatbelt compliance" in results[0]["text"]


def test_result_contains_similarity_score():
    knowledge_base = SafetyKnowledgeBase()
    results = knowledge_base.search("helmet safety")
    assert results
    assert "similarity" in results[0]
    assert isinstance(results[0]["similarity"], float)
