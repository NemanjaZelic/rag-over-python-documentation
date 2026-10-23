#!/usr/bin/env python3

import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import time

import faiss
import numpy as np
from docutils import nodes
from docutils.core import publish_doctree
from docutils.parsers.rst import Directive, directives, roles
import onnxruntime as ort
import tiktoken
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import EMBEDDING_MODEL, EMBEDDING_REVISION, GOLDEN, fts_query, rrf, snapshot


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/experiments"
SOURCES = [
    ("cpython-docs", "https://github.com/python/cpython", "7cd63cddade4c3e1de17c3443695afe1f41d8495", "Doc", {".rst", ".txt"}),
    ("python-peps", "https://github.com/python/peps", "24419b92ae550bf2878716f57c257cba00d3c1a1", "peps", {".rst"}),
]
CONFIGS = [(256, 32), (512, 64), (768, 96)]
DIRECTIVE_RE = re.compile(r"^\s*\.\.\s+([\w:-]+)::", re.MULTILINE)
ROLE_RE = re.compile(r":([\w:.-]+):`")


class AnyOptions(dict):
    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        return lambda value: value


class PreserveDirective(Directive):
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    has_content = True
    option_spec = AnyOptions()

    def run(self):
        container = nodes.container()
        if self.arguments:
            container += nodes.paragraph(text=self.arguments[0])
        self.state.nested_parse(self.content, self.content_offset, container)
        return [container]


def preserve_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    return [nodes.literal(rawtext, text)], []


def documents():
    records = []
    unknown_directives, unknown_roles = set(), set()
    for source_id, repository, commit, source_root, extensions in SOURCES:
        base = ROOT / "data/raw/python-official" / source_id
        for path in sorted(item for item in base.rglob("*") if item.suffix.lower() in extensions):
            text = path.read_text(encoding="utf-8")
            unknown_directives.update(DIRECTIVE_RE.findall(text))
            unknown_roles.update(ROLE_RE.findall(text))
            relative = path.relative_to(base).as_posix()
            records.append((source_id, repository, commit, source_root, relative, path, text))
    for name in unknown_directives:
        if name.lower() not in directives._directive_registry:
            directives.register_directive(name, PreserveDirective)
    for name in unknown_roles:
        if name.lower() not in roles._role_registry:
            roles.register_local_role(name, preserve_role)
    return records


ATOMIC = {"paragraph", "literal_block", "doctest_block", "bullet_list", "enumerated_list", "definition_list", "field_list", "option_list", "table", "line_block", "block_quote"}


def parse_document(record):
    source_id, repository, commit, source_root, relative, path, text = record
    tree = publish_doctree(text, source_path=str(path), settings_overrides={"report_level": 5, "halt_level": 6, "warning_stream": io.StringIO(), "file_insertion_enabled": False, "raw_enabled": False, "doctitle_xform": False})
    blocks = []

    def walk(parent, section_path):
        for child in parent.children:
            if child.tagname == "section":
                title = next((item.astext().strip() for item in child.children if item.tagname == "title"), "")
                walk(child, [*section_path, title] if title else section_path)
            elif child.tagname in {"title", "system_message", "target", "comment", "raw", "transition"}:
                continue
            elif child.tagname in ATOMIC:
                value = child.astext().strip()
                if value:
                    blocks.append({"text": value, "section_path": section_path, "kind": "code" if child.tagname in {"literal_block", "doctest_block"} else "text", "line": child.line})
            else:
                walk(child, section_path)

    walk(tree, [])
    pep_title = re.search(r"(?m)^Title:\s*(.+)$", text)
    first_title = next((block["section_path"][0] for block in blocks if block["section_path"]), "")
    title = pep_title.group(1).strip() if pep_title else first_title or path.stem
    return {
        "document_id": f"python-official/{source_id}/{relative}",
        "title": title,
        "source_url": f"{repository}/blob/{commit}/{source_root}/{relative}",
        "blocks": blocks,
    }


def token_windows(text, encoding, size, overlap):
    token_ids = encoding.encode_ordinary(text)
    start = 0
    while start < len(token_ids):
        end = min(start + size, len(token_ids))
        yield encoding.decode(token_ids[start:end])
        if end == len(token_ids):
            break
        start = end - overlap


def make_chunks(parsed, size, overlap):
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks = []

    def emit(document, section_path, parts, partial=False):
        text = "\n\n".join(parts).strip()
        if not text:
            return
        identity = f"docutils-structural:{size}:{overlap}:{document['document_id']}:{len(chunks)}:{text}"
        chunks.append({
            "chunk_id": "chunk-" + hashlib.sha256(identity.encode()).hexdigest(),
            "document_id": document["document_id"],
            "title": document["title"],
            "section_path": section_path,
            "text": text,
            "token_count": len(encoding.encode_ordinary(text)),
            "contains_partial_block": partial,
            "source_url": document["source_url"],
        })

    for document in parsed:
        current, current_path = [], None
        for block in document["blocks"]:
            path = block["section_path"]
            block_tokens = len(encoding.encode_ordinary(block["text"]))
            if block_tokens > size:
                if current:
                    emit(document, current_path, current)
                    current = []
                for part in token_windows(block["text"], encoding, size, overlap):
                    emit(document, path, [part], True)
                current_path = path
                continue
            candidate = "\n\n".join([*current, block["text"]])
            changed_section = current and path != current_path
            if current and (changed_section or len(encoding.encode_ordinary(candidate)) > size):
                old = current
                emit(document, current_path, old)
                current = []
                if not changed_section:
                    used = 0
                    for part in reversed(old):
                        count = len(encoding.encode_ordinary(part))
                        if used + count > overlap:
                            break
                        current.insert(0, part)
                        used += count
            if not current:
                current_path = path
            current.append(block["text"])
        if current:
            emit(document, current_path, current)
    return chunks


def embedding_text(chunk):
    context = " > ".join(chunk["section_path"])
    prefix = "\n".join(value for value in (chunk["title"], context) if value)
    return f"{prefix}\n\n{chunk['text']}" if prefix else chunk["text"]


def runtime():
    model = snapshot(EMBEDDING_MODEL, EMBEDDING_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    session = ort.InferenceSession(str(model / "model-w-mean-pooling.onnx"), providers=["CPUExecutionProvider"])
    return tokenizer, session, {item.name for item in session.get_inputs()}


def embed(texts, tokenizer, session, names):
    lengths = [len(value) for value in tokenizer(texts, padding=False)["input_ids"]]
    vectors = np.empty((len(texts), 512), dtype="float32")
    order = sorted(range(len(texts)), key=lengths.__getitem__, reverse=True)
    position = 0
    while position < len(order):
        longest = lengths[order[position]]
        size = min(64, max(1, 20_000_000 // (longest * longest)), len(order) - position)
        indices = order[position:position + size]
        inputs = tokenizer([texts[index] for index in indices], padding=True, return_tensors="np")
        batch = session.run(None, {key: value for key, value in inputs.items() if key in names})[0]
        batch /= np.linalg.norm(batch, axis=1, keepdims=True)
        vectors[indices] = batch.astype("float32")
        position += size
    return vectors


def build_variant(chunks, size, overlap, tokenizer, session, names):
    directory = OUTPUT / f"docutils-structural-t{size}-o{overlap}-v1"
    directory.mkdir(parents=True, exist_ok=True)
    chunks_path = directory / "chunks.jsonl"
    index_path = directory / "index.faiss"
    database_path = directory / "index.sqlite3"
    timing_path = directory / "timing.json"
    timing = {}
    if not chunks_path.exists():
        with chunks_path.open("w", encoding="utf-8") as stream:
            for chunk in chunks:
                stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    if not index_path.exists():
        started = time.perf_counter()
        vectors = embed([embedding_text(chunk) for chunk in chunks], tokenizer, session, names)
        index = faiss.IndexFlatIP(512)
        index.add(vectors)
        faiss.write_index(index, str(index_path))
        timing["embedding_and_faiss_seconds"] = time.perf_counter() - started
    if not database_path.exists():
        started = time.perf_counter()
        connection = sqlite3.connect(database_path)
        connection.execute("CREATE VIRTUAL TABLE chunks USING fts5(text, title, section_path, source_row UNINDEXED, tokenize='porter unicode61 remove_diacritics 2')")
        connection.executemany("INSERT INTO chunks VALUES(?,?,?,?)", ((chunk["text"], chunk["title"], " > ".join(chunk["section_path"]), index) for index, chunk in enumerate(chunks)))
        connection.commit()
        connection.close()
        timing["bm25_seconds"] = time.perf_counter() - started
    if timing:
        timing["measurement"] = "perf_counter"
        timing["measured_build_seconds"] = sum(value for value in timing.values() if isinstance(value, float))
        timing_path.write_text(json.dumps(timing, indent=2) + "\n")
    elif not timing_path.exists():
        timing_path.write_text(json.dumps({
            "measurement": "artifact_timestamp_span",
            "initial_artifact_span_seconds": max(index_path.stat().st_mtime, database_path.stat().st_mtime) - chunks_path.stat().st_mtime,
            "note": "Upper-bound wall span from the first run; may include orchestration or interruption time. Future clean builds use perf_counter."
        }, indent=2) + "\n")
    return directory


def load_chunks(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def rank(results, relevant):
    return next((position for position, item in enumerate(results, 1) if item["document_id"] in relevant), None)


def metrics(values):
    count = len(values)
    return {"recall@1": sum(value == 1 for value in values) / count, "recall@5": sum(value is not None and value <= 5 for value in values) / count, "mrr": sum(1 / value for value in values if value) / count}


def evaluate_variant(directory, chunks, questions, query_vectors):
    index = faiss.read_index(str(directory / "index.faiss"))
    connection = sqlite3.connect(f"file:{directory / 'index.sqlite3'}?mode=ro", uri=True)
    ranks = {name: [] for name in ("dense", "bm25", "hybrid")}
    latencies = {name: [] for name in ("dense", "bm25", "hybrid")}
    for query_index, question in enumerate(questions):
        relevant = set(question["relevant_document_ids"])
        hybrid_started = time.perf_counter()
        started = time.perf_counter()
        scores, positions = index.search(query_vectors[query_index:query_index + 1], 60)
        latencies["dense"].append((time.perf_counter() - started) * 1000)
        dense = [{**chunks[position], "score": float(score)} for position, score in zip(positions[0], scores[0])]
        expression = fts_query(question["question"])
        started = time.perf_counter()
        found = connection.execute("SELECT source_row, bm25(chunks, 1.0, 2.0, 1.2, 0.0) FROM chunks WHERE chunks MATCH ? ORDER BY 2 LIMIT 60", (expression,)).fetchall() if expression else []
        latencies["bm25"].append((time.perf_counter() - started) * 1000)
        bm25 = [{**chunks[position], "score": -float(score)} for position, score in found]
        hybrid = rrf(dense, bm25)
        latencies["hybrid"].append((time.perf_counter() - hybrid_started) * 1000)
        for name, results in (("dense", dense), ("bm25", bm25), ("hybrid", hybrid)):
            ranks[name].append(rank(results, relevant))
    connection.close()
    return {
        "metrics": {name: metrics(values) for name, values in ranks.items()},
        "retrieval_latency_ms": {name: {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))} for name, values in latencies.items()},
    }


def save_figure(figure, name):
    for extension in ("png", "svg"):
        figure.savefig(OUTPUT / f"{name}.{extension}", dpi=220, bbox_inches="tight", facecolor="#f8fafc")


def charts(results):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / "matplotlib-cache"))
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    sizes = [size for size, _ in CONFIGS]
    colors = {"dense": "#2563EB", "bm25": "#EA580C", "hybrid": "#059669"}
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    figure.patch.set_facecolor("#f8fafc")
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.76, wspace=0.20)
    for axis in axes:
        axis.set_facecolor("#f8fafc")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xticks(sizes)
        axis.set_xlabel("Maksimalna veličina chunka (tokeni)")
    for metric, label, marker in (("recall@1", "Recall@1", "o"), ("recall@5", "Recall@5", "s"), ("mrr", "MRR", "D")):
        values = [results[str(size)]["metrics"]["hybrid"][metric] for size in sizes]
        axes[0].plot(sizes, values, marker=marker, linewidth=2.6, markersize=7, label=label)
        for x, value in zip(sizes, values):
            axes[0].annotate(f"{value:.3f}", (x, value), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    axes[0].set_title("Hybrid retrieval", loc="left", weight="bold")
    axes[0].set_ylabel("Efektivnost")
    axes[0].set_ylim(0.68, 1.0)
    axes[0].legend(frameon=False, ncol=3, loc="lower left")
    for name in ("dense", "bm25", "hybrid"):
        values = [results[str(size)]["metrics"][name]["mrr"] for size in sizes]
        axes[1].plot(sizes, values, marker="o", linewidth=2.6, markersize=7, label=name.upper(), color=colors[name])
        for x, value in zip(sizes, values):
            axes[1].annotate(f"{value:.3f}", (x, value), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color=colors[name])
    axes[1].set_title("MRR po retrieveru", loc="left", weight="bold")
    axes[1].set_ylabel("MRR")
    axes[1].set_ylim(0.45, 0.92)
    axes[1].legend(frameon=False, ncol=3, loc="lower left")
    figure.suptitle("Efekat veličine strukturnog chunka", fontsize=17, weight="bold", x=0.02, y=0.97, ha="left")
    figure.text(0.02, 0.895, "Docutils parser · Jina embeddings · BM25 + dense RRF · 16 odgovorivih pitanja", fontsize=10.5, color="#475569")
    save_figure(figure, "chunk_size_effectiveness")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.5))
    figure.patch.set_facecolor("#f8fafc")
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.14, top=0.76, wspace=0.25)
    for axis in axes:
        axis.set_facecolor("#f8fafc")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xticks(sizes)
        axis.set_xlabel("Maksimalna veličina chunka (tokeni)")
    counts = [results[str(size)]["chunks"] for size in sizes]
    bars = axes[0].bar(sizes, counts, width=115, color=["#60A5FA", "#3B82F6", "#1D4ED8"])
    axes[0].bar_label(bars, labels=[f"{value:,}".replace(",", ".") for value in counts], padding=4, fontsize=10)
    axes[0].set_title("Veličina indeksa", loc="left", weight="bold")
    axes[0].set_ylabel("Broj chunkova")
    build_minutes = []
    for size in sizes:
        timing = results[str(size)]["build_timing"]
        seconds = timing.get("measured_build_seconds", timing.get("initial_artifact_span_seconds"))
        build_minutes.append(seconds / 60)
    bars = axes[1].bar(sizes, build_minutes, width=115, color=["#FBBF24", "#F59E0B", "#D97706"])
    axes[1].bar_label(bars, labels=[f"{value:.1f} min" for value in build_minutes], padding=4, fontsize=10)
    axes[1].set_title("Prva izgradnja", loc="left", weight="bold")
    axes[1].set_ylabel("Minuti")
    for percentile, marker in (("p50", "o"), ("p95", "s")):
        values = [results[str(size)]["retrieval_latency_ms"]["hybrid"][percentile] for size in sizes]
        axes[2].plot(sizes, values, marker=marker, linewidth=2.6, markersize=7, label=percentile.upper())
        for x, value in zip(sizes, values):
            axes[2].annotate(f"{value:.2f} ms", (x, value), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    axes[2].set_title("Hybrid retrieval latencija", loc="left", weight="bold")
    axes[2].set_ylabel("Milisekunde po upitu")
    axes[2].legend(frameon=False, ncol=2)
    figure.suptitle("Cena finije granulacije", fontsize=17, weight="bold", x=0.02, y=0.97, ha="left")
    figure.text(0.02, 0.895, "Build vremena su prvi zabeleženi wall-clock rasponi; retrieval = FAISS + BM25 + RRF bez query embeddinga", fontsize=10.5, color="#475569")
    save_figure(figure, "chunk_size_runtime")
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    variants = {}
    if all((OUTPUT / f"docutils-structural-t{size}-o{overlap}-v1/chunks.jsonl").exists() for size, overlap in CONFIGS):
        for size, overlap in CONFIGS:
            variants[size] = load_chunks(OUTPUT / f"docutils-structural-t{size}-o{overlap}-v1/chunks.jsonl")
    else:
        parsed = [parse_document(record) for record in documents()]
        for size, overlap in CONFIGS:
            variants[size] = make_chunks(parsed, size, overlap)
    tokenizer, session, names = runtime()
    directories = {size: build_variant(variants[size], size, overlap, tokenizer, session, names) for size, overlap in CONFIGS}
    with GOLDEN.open(encoding="utf-8") as stream:
        questions = [json.loads(line) for line in stream if line.strip() and json.loads(line)["answerable"]]
    query_vectors = embed([question["question"] for question in questions], tokenizer, session, names)
    results = {}
    for size, _ in CONFIGS:
        evaluation = evaluate_variant(directories[size], variants[size], questions, query_vectors)
        results[str(size)] = {
            "chunks": len(variants[size]),
            "index_bytes": (directories[size] / "index.faiss").stat().st_size,
            "metrics": evaluation["metrics"],
            "retrieval_latency_ms": evaluation["retrieval_latency_ms"],
            "build_timing": json.loads((directories[size] / "timing.json").read_text()),
        }
    (OUTPUT / "chunk_size_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    charts(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
