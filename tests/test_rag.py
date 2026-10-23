import unittest

from rag import fts_query, lexical_terms, rrf


def row(chunk_id, score=1):
    return {"chunk_id": chunk_id, "score": score}


class RagTests(unittest.TestCase):
    def test_terms(self):
        self.assertEqual(lexical_terms("PEP8 recommends 4 spaces"), ["pep8", "pep", "8", "recommends", "4", "spaces"])

    def test_fts(self):
        self.assertEqual(fts_query("PEP 8"), '"pep" OR "8"')

    def test_empty_fts(self):
        self.assertEqual(fts_query("___"), "")

    def test_rrf_agreement(self):
        result = rrf([row("a"), row("b")], [row("b"), row("c")])
        self.assertEqual([item["chunk_id"] for item in result], ["b", "a", "c"])

    def test_rrf_ignores_score_scale(self):
        first = rrf([row("a", 0.1)], [row("b", 1000)])
        second = rrf([row("a", 1000)], [row("b", 0.1)])
        self.assertEqual([item["chunk_id"] for item in first], [item["chunk_id"] for item in second])

    def test_rrf_ranks(self):
        result = rrf([row("a"), row("b")], [row("b")])
        self.assertEqual(result[0]["dense_rank"], 2)
        self.assertEqual(result[0]["bm25_rank"], 1)


if __name__ == "__main__":
    unittest.main()
