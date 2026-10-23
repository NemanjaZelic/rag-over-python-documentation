#!/usr/bin/env python3

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent
CHUNKS = ROOT / "data/chunks/python-official-t512-o64-v1/chunks.jsonl"
DENSE = ROOT / "data/indexes/python-official-t512-o64-jina-small-v1"
SPARSE = ROOT / "data/indexes/python-official-t512-o64-bm25-v1"
SOURCES = [
    ("cpython-docs", "https://github.com/python/cpython", "7cd63cddade4c3e1de17c3443695afe1f41d8495", "Doc", {".rst", ".txt"}),
    ("python-peps", "https://github.com/python/peps", "24419b92ae550bf2878716f57c257cba00d3c1a1", "peps", {".rst"}),
]


def model_cache() -> Path:
    root = ROOT / "data/model-cache"
    os.environ.setdefault("HF_HOME", str(root / "home"))
    os.environ.setdefault("HF_MODULES_CACHE", str(root / "modules"))
    os.environ.setdefault("HF_XET_CACHE", str(root / "xet"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / "xdg"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return root / "hub"


def download() -> None:
    raw = ROOT / "data/raw/python-official"
    if all((raw / name).exists() for name, *_ in SOURCES):
        return
    raw.mkdir(parents=True, exist_ok=True)
    for name, repository, commit, source_root, extensions in SOURCES:
        target = raw / name
        if target.exists():
            continue
        target.mkdir()
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive_file:
            request = urllib.request.Request(f"{repository}/archive/{commit}.tar.gz", headers={"User-Agent": "rag"})
            with urllib.request.urlopen(request) as response:
                archive_file.write(response.read())
                archive_file.flush()
            with tarfile.open(archive_file.name, "r:gz") as archive:
                for member in archive.getmembers():
                    parts = Path(member.name).parts
                    if not member.isfile() or source_root not in parts:
                        continue
                    index = parts.index(source_root)
                    relative = Path(*parts[index + 1:])
                    if relative.suffix.lower() not in extensions or ".." in relative.parts:
                        continue
                    destination = target / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, destination.open("wb") as output:
                        output.write(source.read())


def build_chunks() -> None:
    if CHUNKS.exists():
        return
    import tiktoken
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / "data/tokenizer-cache"))
    tokenizer = tiktoken.get_encoding("cl100k_base")
    CHUNKS.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS.open("w", encoding="utf-8") as output:
        for name, repository, commit, source_root, extensions in SOURCES:
            base = ROOT / "data/raw/python-official" / name
            for path in sorted(file for file in base.rglob("*") if file.suffix.lower() in extensions):
                text = path.read_text(encoding="utf-8")
                tokens = tokenizer.encode_ordinary(text)
                start = 0
                chunk_index = 0
                while start < len(tokens):
                    end = min(start + 512, len(tokens))
                    value = tokenizer.decode(tokens[start:end])
                    relative = path.relative_to(base).as_posix()
                    document_id = f"python-official/{name}/{relative}"
                    digest = hashlib.sha256(f"{document_id}:{chunk_index}:{value}".encode()).hexdigest()
                    row = {"chunk_id": digest[:24], "document_id": document_id, "title": path.stem, "section_paths": [], "text": value, "text_sha256": hashlib.sha256(value.encode()).hexdigest(), "source_spans": [{"source_url": f"{repository}/blob/{commit}/{source_root}/{relative}"}]}
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if end == len(tokens):
                        break
                    start = end - 64
                    chunk_index += 1


def embedding_snapshot() -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download("jinaai/jina-embeddings-v2-small-en", revision="44e7d1d6caec8c883c2d4b207588504d519788d0", cache_dir=model_cache()))


def build_dense() -> None:
    if (DENSE / "index.faiss").exists() and (DENSE / "rows.jsonl").exists():
        return
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer
    texts, records, offsets = [], [], []
    with CHUNKS.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            row = json.loads(line)
            offsets.append(offset)
            records.append(row)
            texts.append(row["text"])
    model = embedding_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    session = ort.InferenceSession(str(model / "model-w-mean-pooling.onnx"), providers=["CPUExecutionProvider"])
    names = {item.name for item in session.get_inputs()}
    lengths = [len(value) for value in tokenizer(texts, padding=False)["input_ids"]]
    if max(lengths) > 4096:
        raise ValueError("Chunk exceeds embedding model limit")
    embeddings = np.empty((len(texts), 512), dtype="float32")
    order = sorted(range(len(texts)), key=lengths.__getitem__, reverse=True)
    position = 0
    while position < len(order):
        longest = lengths[order[position]]
        size = min(32, max(1, 10_000_000 // (longest * longest)), len(order) - position)
        indices = order[position:position + size]
        inputs = tokenizer([texts[index] for index in indices], padding=True, return_tensors="np")
        batch = session.run(None, {key: value for key, value in inputs.items() if key in names})[0]
        batch /= np.linalg.norm(batch, axis=1, keepdims=True)
        embeddings[indices] = batch.astype("float32")
        position += size
    DENSE.mkdir(parents=True, exist_ok=True)
    temporary = DENSE / "embeddings.tmp.npy"
    np.save(temporary, embeddings)
    subprocess.run([sys.executable, str(ROOT / "build.py"), "_faiss", str(temporary)], check=True)
    temporary.unlink()
    with (DENSE / "rows.jsonl").open("w", encoding="utf-8") as output:
        for index, (record, offset) in enumerate(zip(records, offsets)):
            output.write(json.dumps({"row": index, "chunk_id": record["chunk_id"], "text_sha256": record["text_sha256"], "chunks_jsonl_byte_offset": offset}) + "\n")


def faiss_build(path: str) -> None:
    import faiss
    import numpy as np
    vectors = np.load(path).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(DENSE / "index.faiss"))


def build_sparse() -> None:
    if (SPARSE / "index.sqlite3").exists():
        return
    SPARSE.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(SPARSE / "index.sqlite3")
    connection.execute("CREATE VIRTUAL TABLE chunks USING fts5(text, title, document_id, section_paths, source_row UNINDEXED, tokenize='porter unicode61 remove_diacritics 2')")
    with CHUNKS.open(encoding="utf-8") as stream:
        batch = []
        for index, line in enumerate(stream):
            row = json.loads(line)
            sections = " | ".join(section for path in row.get("section_paths", []) for section in path)
            batch.append((row["text"], row["title"], row["document_id"], sections, index))
            if len(batch) == 256:
                connection.executemany("INSERT INTO chunks VALUES(?,?,?,?,?)", batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO chunks VALUES(?,?,?,?,?)", batch)
    connection.commit()
    connection.close()


def prepare_generator() -> None:
    from huggingface_hub import snapshot_download
    snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775", cache_dir=model_cache())


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "_faiss":
        faiss_build(sys.argv[2])
        return
    download()
    build_chunks()
    build_dense()
    build_sparse()
    embedding_snapshot()
    prepare_generator()


if __name__ == "__main__":
    main()
