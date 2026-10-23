#!/usr/bin/env python3
"""Deterministic, Unicode-safe token windows with exact normalized-text provenance."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.0.0'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def tokenizer(config):
    os.environ['TIKTOKEN_CACHE_DIR'] = str(ROOT/'data/tokenizer-cache')
    import tiktoken
    if version('tiktoken') != config['tiktoken_version']:
        raise ValueError('Install the exact tiktoken version in requirements.txt')
    return tiktoken.get_encoding(config['tokenizer'])


def validate_config(config):
    if config['algorithm'] != 'fixed_token_window':
        raise ValueError('Unsupported chunking algorithm')
    size, overlap = config['max_tokens'], config['overlap_tokens']
    if type(size) is not int or type(overlap) is not int or not 8 <= size or not 0 <= overlap < size:
        raise ValueError('Require integer max_tokens >= 8 and 0 <= overlap_tokens < max_tokens')
    if config['include_title_in_text'] is not False or config['retrievable_only'] is not True:
        raise ValueError('This baseline keeps titles in metadata and filters retrievable documents')


def windows(text, enc, size, overlap):
    """Return document-token windows; re-encode each window to enforce the cap.

    BPE tokens can end inside a UTF-8 character. Only boundaries shared with
    Unicode character boundaries are used. No replacement characters or gaps
    are introduced. Overlap is targeted in the original document tokenization.
    """
    tokens = enc.encode_ordinary(text)
    if not tokens:
        return [], 0
    token_bytes = [0]
    for token in tokens:
        token_bytes.append(token_bytes[-1] + len(enc.decode_single_token_bytes(token)))
    wanted = set(token_bytes)
    char_at_byte = {0: 0}
    byte_position = 0
    for index, char in enumerate(text, 1):
        byte_position += len(char.encode('utf-8'))
        if byte_position in wanted:
            char_at_byte[byte_position] = index
    valid = [i for i, b in enumerate(token_bytes) if b in char_at_byte]
    char_at_token = {i: char_at_byte[token_bytes[i]] for i in valid}
    result, start = [], 0
    while start < len(tokens):
        end_index = bisect_right(valid, min(start+size, len(tokens)))-1
        while valid[end_index] > start:
            end = valid[end_index]
            first, last = char_at_token[start], char_at_token[end]
            count = len(enc.encode_ordinary(text[first:last]))
            if count <= size:
                break
            end_index -= 1
        else:
            raise ValueError('No complete Unicode character fits the token budget')
        if result and end <= result[-1]['document_token_end']:
            # With near-total overlap and multibyte text, retokenization can
            # otherwise yield a redundant window that adds no new content.
            start = valid[bisect_right(valid, start)]
            continue
        result.append({'start_char': first, 'end_char': last,
                       'document_token_start': start, 'document_token_end': end,
                       'token_count': count})
        if end == len(tokens):
            break
        start = valid[bisect_left(valid, max(start+1, end-overlap))]
    return result, len(tokens)


def chunks_for_document(doc, enc, config, config_id):
    spans, total_tokens = windows(doc['text'], enc, config['max_tokens'], config['overlap_tokens'])
    blocks = doc['blocks']
    ends = [b['end_char'] for b in blocks]
    records, previous = [], None
    for ordinal, window in enumerate(spans):
        start, end = window['start_char'], window['end_char']
        text = doc['text'][start:end]
        sources, section_paths = [], []
        partial_code = False
        block_index = bisect_right(ends, start)
        for b in blocks[block_index:]:
            if b['start_char'] >= end:
                break
            lo, hi = max(start, b['start_char']), min(end, b['end_char'])
            if lo >= hi:
                continue
            full = lo == b['start_char'] and hi == b['end_char']
            partial_code |= b['kind']=='code' and not full
            if b['section_path'] not in section_paths:
                section_paths.append(b['section_path'])
            source = {k: b[k] for k in ('block_id', 'kind', 'source_id', 'source_repo_path', 'source_commit', 'source_sha256', 'source_url', 'source_start_line', 'source_end_line', 'section_path')}
            source.update({'document_start_char': lo, 'document_end_char': hi,
                           'chunk_start_char': lo-start, 'chunk_end_char': hi-start,
                           'whole_block': full, 'source_line_precision': 'enclosing_normalized_block'})
            for key in ('included_via', 'derived_from', 'generated_by', 'selected_source_lines'):
                if key in b:
                    source[key] = b[key]
            sources.append(source)
        overlap_end = previous['end_char'] if previous else start
        overlap_count = len(enc.encode_ordinary(doc['text'][start:overlap_end])) if overlap_end > start else 0
        identity = {'config_id': config_id, 'document_id': doc['document_id'],
                    'document_text_sha256': doc['text_sha256'], 'start_char': start, 'end_char': end}
        records.append({'chunk_id': 'chunk-' + sha(canonical(identity)), 'chunk_index': ordinal,
                        'config_id': config_id, 'chunker_version': VERSION,
                        'document_id': doc['document_id'], 'document_text_sha256': doc['text_sha256'],
                        'title': doc['title'], 'metadata': doc['metadata'],
                        'text': text, 'text_sha256': sha(text.encode()), **window,
                        'tokenizer': config['tokenizer'], 'overlap_with_previous_tokens': overlap_count,
                        'overlap_in_document_tokens': previous['document_token_end']-window['document_token_start'] if previous else 0,
                        'section_paths': section_paths, 'crosses_section_boundary': len(section_paths)>1,
                        'contains_partial_code_block': partial_code, 'source_spans': sources})
        previous = window
    return records, total_tokens


def distribution(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    return {'count': len(values), 'min': values[0], 'p50': values[math.ceil(len(values)*.5)-1],
            'p95': values[math.ceil(len(values)*.95)-1], 'max': values[-1],
            'mean': round(sum(values)/len(values), 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT/'config/chunking.json')
    ap.add_argument('--input', type=Path, default=ROOT/'data/processed/python-official-v1.1.1/documents.jsonl')
    ap.add_argument('--output', type=Path, default=ROOT/'data/chunks/python-official-t512-o64-v1')
    ap.add_argument('--max-tokens', type=int)
    ap.add_argument('--overlap-tokens', type=int)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Output already exists; choose a new --output directory')
    config = json.loads(args.config.read_text())
    for key in ('max_tokens', 'overlap_tokens'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    validate_config(config)
    if file_sha(args.input) != config['input_documents_sha256']:
        raise ValueError('Normalized corpus does not match locked input hash')
    config_id = sha(canonical({'config': config, 'chunker_version': VERSION}))
    enc = tokenizer(config)
    counts, by_source = Counter(), Counter()
    lengths, overlaps, document_lengths, per_doc = [], [], [], []
    excluded, examples, seen = [], [], set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix='.chunking-') as tmp:
        stage = Path(tmp)
        with args.input.open(encoding='utf-8') as stream, (stage/'chunks.jsonl').open('w', encoding='utf-8') as out:
            for row in stream:
                doc = json.loads(row)
                counts['input_documents'] += 1
                if doc['retrievable'] is not True:
                    excluded.append({'document_id': doc['document_id'], 'reason': doc.get('exclusion_reason', '')})
                    continue
                if sha(doc['text'].encode()) != doc['text_sha256']:
                    raise ValueError('Document text checksum mismatch')
                if any(b['kind']=='unresolved' for b in doc['blocks']):
                    raise ValueError('Unresolved content must be reviewed before chunking')
                chunks, n_tokens = chunks_for_document(doc, enc, config, config_id)
                counts['included_documents'] += 1
                counts['document_tokens_without_overlap'] += n_tokens
                document_lengths.append(n_tokens)
                per_doc.append(len(chunks))
                for chunk in chunks:
                    if chunk['chunk_id'] in seen:
                        raise ValueError('Duplicate chunk identity')
                    seen.add(chunk['chunk_id'])
                    out.write(json.dumps(chunk, ensure_ascii=False)+'\n')
                    counts['chunks'] += 1
                    counts['chunk_tokens_with_overlap'] += chunk['token_count']
                    counts['chunks_below_64_tokens'] += chunk['token_count'] < 64
                    counts['chunks_crossing_sections'] += chunk['crosses_section_boundary']
                    counts['chunks_with_partial_code'] += chunk['contains_partial_code_block']
                    by_source[doc['source_id']] += 1
                    lengths.append(chunk['token_count'])
                    if chunk['chunk_index']:
                        overlaps.append(chunk['overlap_with_previous_tokens'])
                    if doc['source_path'] in {'cpython-docs/tutorial/controlflow.rst', 'python-peps/pep-0008.rst'} and chunk['chunk_index']<2:
                        examples.append(chunk)
        counts['excluded_documents'] = len(excluded)
        report = {'schema_version': 1, 'chunker_version': VERSION, 'config_id': config_id,
                  'config': config, 'chunker_sha256': file_sha(Path(__file__)),
                  'python_version': platform.python_version(), 'tiktoken_version': version('tiktoken'),
                  'tokenizer_cache_sha256': sorted(file_sha(p) for p in (ROOT/'data/tokenizer-cache').iterdir() if p.is_file()),
                  'counts': dict(counts), 'chunks_by_source': dict(by_source),
                  'chunk_token_distribution': distribution(lengths),
                  'overlap_token_distribution': distribution(overlaps),
                  'document_token_distribution': distribution(document_lengths),
                  'chunks_per_document': distribution(per_doc), 'excluded_documents': excluded,
                  'chunks_sha256': file_sha(stage/'chunks.jsonl'),
                  'limitations': ['Fixed windows can split sentences, sections and code',
                                  'Token counts use cl100k_base, not a yet-selected embedding model',
                                  'Overlap targets document-token positions; Unicode alignment and re-encoding can change measured overlap',
                                  'Source line ranges enclose original blocks; exact intersections refer to normalized text',
                                  'No embedding, retrieval, deduplication or RAG quality evaluation performed']}
        (stage/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        (stage/'examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        shutil.move(str(stage), str(args.output))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
