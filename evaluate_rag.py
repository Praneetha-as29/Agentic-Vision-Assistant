
from rag import SafetyKnowledgeBase

TEST_CASES = [
    {
        "question": "Which road users should be identified in a traffic scene?",
        "expected": "Identify visible road users"
    },
    {
        "question": "Can detecting a vehicle prove a traffic violation?",
        "expected": "A vehicle or person detection is not sufficient evidence"
    },
    {
        "question": "What should I check before declaring a seatbelt violation?",
        "expected": "Seatbelt compliance should only be assessed"
    },
    {
        "question": "Does seeing a helmet prove that it meets safety standards?",
        "expected": "image detection alone does not prove"
    },
    {
        "question": "How should uncertain visual evidence be handled?",
        "expected": "report uncertainty instead of making a definite claim"
    },
    {
        "question": "Does a high object detection confidence prove a traffic violation?",
        "expected": "not the probability that a person is following traffic rules"
    },
    {
        "question": "What do bounding-box coordinates describe?",
        "expected": "Bounding-box coordinates describe the detected object's location"
    },
    {
        "question": "Can an unclear image support a definite conclusion?",
        "expected": "report uncertainty instead of making a definite claim"
    },
    {
        "question": "What does object detection tell us about road users?",
        "expected": "Identify visible road users"
    },
    {
        "question": "When should seatbelt compliance be assessed?",
        "expected": "Seatbelt compliance should only be assessed"
    }
]


def main():
    knowledge_base = SafetyKnowledgeBase()
    hits = 0

    print("\nRAG RETRIEVAL EVALUATION")
    print("=" * 60)

    for index, test_case in enumerate(TEST_CASES, start=1):
        results = knowledge_base.search(test_case["question"])
        expected = test_case["expected"].lower()

        found = any(
            expected in result["text"].lower()
            for result in results[:3]
        )

        hits += int(found)
        print(f"\nTest {index}: {'PASS' if found else 'MISS'}")
        print(f"Question: {test_case['question']}")
        print(f"Expected guideline phrase: {test_case['expected']}")

        for rank, result in enumerate(results[:3], start=1):
            print(
                f"  {rank}. {result['text']} "
                f"(similarity: {result['similarity']:.4f})"
            )

    recall_at_3 = hits / len(TEST_CASES) * 100

    print("\n" + "=" * 60)
    print(f"Questions evaluated: {len(TEST_CASES)}")
    print(f"Expected guidelines retrieved in top 3: {hits}")
    print(f"Recall@3: {recall_at_3:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
