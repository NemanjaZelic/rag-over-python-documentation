import hashlib
import json
from pathlib import Path
import sqlite3

import faiss


ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data/chunks/python-official-t512-o64-v1/chunks.jsonl"
DENSE = ROOT / "data/indexes/python-official-t512-o64-jina-small-v1"
BM25 = ROOT / "data/indexes/python-official-t512-o64-bm25-v1/index.sqlite3"


def main():
    with (DENSE / "rows.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    with CHUNKS.open("rb") as stream:
        for index, row in enumerate(rows):
            assert row["row"] == index
            stream.seek(row["chunks_jsonl_byte_offset"])
            value = json.loads(stream.readline())
            assert value["chunk_id"] == row["chunk_id"]
            assert hashlib.sha256(value["text"].encode()).hexdigest() == row["text_sha256"]
    index = faiss.read_index(str(DENSE / "index.faiss"))
    assert index.ntotal == len(rows)
    assert index.d == 512
    connection = sqlite3.connect(f"file:{BM25}?mode=ro", uri=True)
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert connection.execute("SELECT count(*) FROM chunks").fetchone()[0] == len(rows)
    connection.close()
    print("OK")


if __name__ == "__main__":
    main()
