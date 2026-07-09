import unittest

from rag import build_prompt


class RagPromptTests(unittest.TestCase):
    def test_prompt_requires_answers_from_retrieved_context_only(self):
        docs = [type("Doc", (), {"page_content": "The PDF says the sky is blue."})()]

        prompt = build_prompt("What color is the sky?", docs)

        self.assertIn("Only use the retrieved context", prompt)
        self.assertIn("Do not answer from outside the document", prompt)


if __name__ == "__main__":
    unittest.main()
