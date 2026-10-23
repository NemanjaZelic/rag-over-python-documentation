#!/usr/bin/env python3
"""Conservative RST source normalization, not a Sphinx renderer.

Local includes are read from hash-verified snapshots; no source code is executed.
Unsupported constructs are preserved and reported rather than silently lost.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tempfile
import textwrap
import shutil
import csv
import io
from html.parser import HTMLParser
from corpus_sources import SnapshotStore, select_lines

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.1.1"
DIRECTIVE = re.compile(r"^(\s*)\.\. ([\w:-]+)::\s*(.*)$")
ROLE = re.compile(r":([\w:.-]+):`([^`]+)`")
SKIP = {"index", "toctree", "contents", "sectnum", "currentmodule", "currentclass", "default-role", "highlight", "tab-width"}
SKIP |= {"tabularcolumns", "c:namespace"}
CODE = {"code", "code-block", "sourcecode", "doctest", "testcode", "testoutput"}
CODE |= {"testsetup", "testcleanup", "grammar-snippet", "parsed-literal"}
KNOWN = {"note", "warning", "important", "tip", "hint", "attention", "caution", "danger", "seealso", "admonition", "versionadded", "versionchanged", "deprecated", "deprecated-removed", "availability", "impl-detail", "implementation-detail", "function", "method", "class", "exception", "attribute", "data", "module", "option", "describe", "decorator", "decoratormethod", "audit-event", "productionlist", "rubric", "only", "list-table", "csv-table", "table", "math", "figure", "image", "topic", "container", "compound"}
KNOWN |= {"soft-deprecated", "versionremoved", "object", "program", "awaitablemethod", "awaitablefunction", "classmethod", "staticmethod", "property", "cmdoption", "opcode", "pdbcommand", "monitoring-event", "envvar", "corresponding-type-slot", "glossary", "sidebar", "epigraph", "pull-quote", "withdrawn", "rejected", "superseded", "canonical-doc", "canonical-pypa-spec", "canonical-typing-spec"}
INCLUDE_OPTIONS = {'start-line', 'end-line', 'start-at', 'start-after', 'end-at', 'end-before', 'lines', 'code', 'literal', 'language', 'linenos', 'lineno-start', 'emphasize-lines', 'name', 'dedent', 'encoding'}
LABELS = {
    'soft-deprecated': 'Soft-deprecated since', 'versionremoved': 'Removed in',
    'opcode': 'Opcode', 'envvar': 'Environment variable', 'classmethod': 'Class method',
    'staticmethod': 'Static method', 'property': 'Property',
    'awaitablemethod': 'Awaitable method', 'awaitablefunction': 'Awaitable function',
    'monitoring-event': 'sys.monitoring.events', 'pdbcommand': 'Debugger command',
    'withdrawn': 'This PEP has been withdrawn.', 'rejected': 'This PEP has been rejected.',
    'superseded': 'This PEP has been superseded by PEP',
    'canonical-doc': 'This PEP is historical. Canonical documentation is now at',
    'canonical-pypa-spec': 'This PEP is historical. Canonical PyPA specification is at',
    'canonical-typing-spec': 'This PEP is historical. Canonical typing specification is at',
}


class VisibleHTML(HTMLParser):
    """Text only: scripts, CSS, comments and DOM behavior are never executed."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ignored = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style'}:
            self.ignored += 1
        if not self.ignored and tag in {'p', 'br', 'div', 'tr', 'text'}:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in {'script', 'style'}:
            self.ignored = max(0, self.ignored-1)
        if not self.ignored and tag in {'p', 'div', 'tr', 'text'}:
            self.parts.append('\n')

    def handle_data(self, data):
        if not self.ignored:
            self.parts.append(data)


def directive_parts(lines, start, end):
    """Options immediately following a directive; retain body line positions."""
    options = {}
    body = start+1
    while body < end:
        if not lines[body].strip():
            body += 1
            continue
        option = re.match(r'^\s+:([\w-]+):\s*(.*)$', lines[body])
        if option:
            options[option[1]] = option[2]
            body += 1
        else:
            break
    return options, body


def generated_content(name, store, sid):
    if name in {'limited-api-list', 'corresponding-type-slot'}:
        path = 'Doc/data/stable_abi.dat'
        text = store.get(sid, path)
        rows = list(csv.DictReader(io.StringIO(text)))
        return rows, store.provenance(sid, path)
    path = 'Include/patchlevel.h'
    text = store.get(sid, path)
    minor = int(re.search(r'#define PY_MINOR_VERSION\s+(\d+)', text)[1])
    content = '\n'.join(f'{(3 << 24) | (m << 16):#x}  /* Py_PACK_VERSION(3,{m}) */'
                        for m in range(minor-5, minor+1))
    return content, store.provenance(sid, path)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inline(value: str) -> str:
    # Protect inline code; a role-looking string within code is literal data.
    def clean(part: str) -> str:
        def replace(match):
            role, body = match.groups()
            explicit = re.fullmatch(r"(.+?)\s+<([^>]+)>", body, flags=re.DOTALL)
            if explicit:
                return explicit.group(1)
            if role == "pep":
                return "PEP " + body
            if role == "rfc":
                return "RFC " + body
            return body.lstrip("!")
        part = ROLE.sub(replace, part)
        return re.sub(r"`([^`]+)\s+<([^>]+)>`__?", r"\1 (\2)", part)
    return "".join(p if p.startswith("``") else clean(p)
                   for p in re.split(r"(``.*?``)", value))


def adornment(line: str) -> bool:
    s = line.strip()
    return len(s) >= 3 and len(set(s)) == 1 and s[0] in "=-~^\"'`:+*#_<>."


def parse(text: str, document_id: str, *, store=None, source_id=None, repo_path=None,
          line_numbers=None, ancestors=(), section_prefix=()) -> dict:
    lines = text.splitlines()
    line_numbers = line_numbers if line_numbers is not None else list(range(1, len(lines)+1))
    origin = store.provenance(source_id, repo_path) if store else {}
    metadata, blocks, sections, warnings = {}, [], [], []
    inclusions, omissions = [], []
    directives = Counter()
    styles, stack = [], []
    title = ""
    i = 0
    if lines and re.match(r"^PEP:\s*\d+", lines[0]):
        key = None
        while i < len(lines) and lines[i].strip():
            m = re.match(r"^([\w-]+):\s*(.*)$", lines[i])
            if m:
                key = m[1]; metadata[key] = m[2]
            elif key:
                metadata[key] += " " + lines[i].strip()
            i += 1
        title = metadata.get("Title", "")

    def emit(kind, content, start, end, **extra):
        content = content.strip("\n")
        if not content.strip():
            return
        blocks.append({"block_id": f"{document_id}:b{len(blocks):05d}",
                       "kind": kind, "text": content,
                       "source_start_line": line_numbers[start], "source_end_line": line_numbers[end-1],
                       "section_path": list(section_prefix) + [s["title"] for s in stack],
                       **origin, **extra})

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1; continue
        # Top-level section titles, including overline + underline style.
        heading = None
        if not line[0].isspace():
            if (adornment(line) and i + 2 < len(lines) and lines[i+1].strip()
                    and lines[i+2].strip() == line.strip()):
                heading = (lines[i+1], "over:" + line[0], 3)
            elif (i + 1 < len(lines) and adornment(lines[i+1])
                  and not adornment(line) and not line.startswith("..")):
                heading = (line, "under:" + lines[i+1].strip()[0], 2)
        if heading:
            name, style, consumed = heading
            name = inline(name)
            if style not in styles:
                styles.append(style)
            level = styles.index(style) + 1
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            section = {"title": name, "level": level, "source_line": line_numbers[i], **origin}
            stack.append(section)
            sections.append({**section, "section_path": list(section_prefix) + [s["title"] for s in stack]})
            if not title:
                title = name
            emit("heading", name, i, i+consumed)
            i += consumed; continue

        match = DIRECTIVE.match(line)
        if match:
            whitespace, name, arg = match.groups()
            name = name.lower()
            indent = len(whitespace)
            directives[name] += 1
            j = i+1
            while j < len(lines) and (not lines[j].strip() or len(lines[j])-len(lines[j].lstrip()) > indent):
                j += 1
            options, body = directive_parts(lines, i, j)
            if name in SKIP:
                omissions.append({'line': line_numbers[i], 'directive': name, 'reason': 'navigation_or_layout', **origin})
                i = j; continue
            if name in CODE:
                content = textwrap.dedent("\n".join(lines[body:j])).rstrip()
                if name == 'parsed-literal':
                    content = inline(content)
                emit("code", content, body, j, directive_name=name, directive_options=options)
                i = j; continue
            if name in {'include', 'literalinclude'} and store:
                event = {'directive': name, 'line': line_numbers[i], 'argument': arg,
                         'options': options, **origin}
                try:
                    unsupported = set(options) - INCLUDE_OPTIONS
                    if unsupported:
                        raise ValueError(f'Unsupported include options: {sorted(unsupported)}')
                    if options.get('encoding', 'utf-8').lower() not in {'utf-8', 'utf8'}:
                        raise ValueError('Only UTF-8 includes supported')
                    target = store.resolve(source_id, repo_path, arg)
                    if (source_id, target) in ancestors + ((source_id, repo_path),):
                        raise ValueError('Recursive include cycle')
                    if len(ancestors) >= 16:
                        raise ValueError('Include nesting limit exceeded')
                    selected, numbers = select_lines(store.get(source_id, target), options)
                    target_origin = store.provenance(source_id, target)
                    event.update({'status': 'resolved', 'target': target_origin,
                                  'selected_lines': numbers})
                    site = {**origin, 'source_start_line': line_numbers[i], 'source_end_line': line_numbers[j-1]}
                    if name == 'literalinclude' or 'code' in options or 'literal' in options:
                        content = '\n'.join(selected)
                        if 'dedent' in options:
                            n = options['dedent']
                            content = textwrap.dedent(content) if not n else '\n'.join(s[int(n):] for s in selected)
                        if content.strip():
                            emit('code', content, i, j, **target_origin,
                                 included_via=[site], selected_source_lines=numbers,
                                 directive_name=name, directive_options=options)
                            blocks[-1]['source_start_line'] = numbers[0]
                            blocks[-1]['source_end_line'] = numbers[-1]
                    else:
                        child = parse('\n'.join(selected), document_id, store=store,
                                      source_id=source_id, repo_path=target, line_numbers=numbers,
                                      ancestors=ancestors + ((source_id, repo_path),),
                                      section_prefix=list(section_prefix)+[s['title'] for s in stack])
                        for block in child['blocks']:
                            block['included_via'] = [site] + block.get('included_via', [])
                        blocks.extend(child['blocks']); sections.extend(child['sections'])
                        warnings.extend(child['warnings']); omissions.extend(child['omissions'])
                        inclusions.extend(child['inclusions']); directives.update(child['directive_counts'])
                    inclusions.append(event)
                except (ValueError, FileNotFoundError, UnicodeError) as error:
                    event.update({'status': 'failed', 'error': str(error)})
                    inclusions.append(event)
                    warnings.append({'line': line_numbers[i], 'type': 'include_error', 'detail': str(error), **origin})
                    emit('unresolved', '\n'.join(lines[i:j]), i, j)
                i = j; continue
            if name == 'raw':
                content = '\n'.join(lines[body:j])
                raw_origin = None
                if 'file' in options:
                    target = store.resolve(source_id, repo_path, options['file']) if store else options['file']
                    if target.endswith('.css'):
                        content = ''
                    elif store:
                        try:
                            content = store.get(source_id, target)
                            raw_origin = store.provenance(source_id, target)
                        except FileNotFoundError:
                            omissions.append({'line': line_numbers[i], 'directive': name, 'reason': 'missing_visual_build_asset', 'target': target, **origin})
                            i = j; continue
                extractor = VisibleHTML(); extractor.feed(content)
                visible = ''.join(extractor.parts).strip()
                if visible:
                    emit('text', visible, i, j, derived_from=[raw_origin] if raw_origin else [])
                else:
                    omissions.append({'line': line_numbers[i], 'directive': name, 'reason': 'html_layout_script_style_or_anchor', **origin})
                i = j; continue
            if name == 'miscnews':
                # The upstream extension also reports unavailable NEWS when the build file is absent.
                omissions.append({'line': line_numbers[i], 'directive': name,
                                  'reason': 'unavailable_generated_NEWS', 'argument': arg, **origin})
                i = j; continue
            if name in {'version-hex-cheatsheet', 'limited-api-list'} and store:
                content, dependency = generated_content(name, store, source_id)
                if isinstance(content, list):
                    content = '\n'.join(f"- {r['name']} ({r['role']})" for r in content)
                emit('code' if name == 'version-hex-cheatsheet' else 'text', content, i, j,
                     generated_by=name, derived_from=[dependency])
                i = j; continue
            if name == 'audit-event-table':
                emit('text', 'Audit events', i, j, generated_by=name)
                i = j; continue
            if name == 'csv-table' and 'file' not in options and 'url' not in options:
                rows = list(csv.reader(io.StringIO(textwrap.dedent('\n'.join(lines[body:j]))), skipinitialspace=True))
                if 'header' in options:
                    rows.insert(0, next(csv.reader([options['header']], skipinitialspace=True)))
                content = ([inline(arg)] if arg else []) + [inline(' | '.join(' '.join(cell.split()) for cell in row)) for row in rows if row]
                emit('table', '\n'.join(content), i, j, directive_name=name, directive_options=options)
                i = j; continue
            if name in {"include", "literalinclude"}:
                warnings.append({"line": i+1, "type": "unresolved_" + name, "argument": arg})
                emit("unresolved", "\n".join(lines[i:j]), i, j)
                i = j; continue
            if name not in KNOWN and not name.startswith(("c:", "py:")):
                warnings.append({"line": i+1, "type": "unsupported_directive", "name": name})
                emit("unresolved", "\n".join(lines[i:j]), i, j)
                i = j; continue
            # Preserve conditions, API signatures, version numbers and captions.
            label = LABELS.get(name, name)
            value = inline(f"{label}: {arg}".rstrip(': '))
            for option in ('header', 'type', 'value', 'annotation', 'platform', 'alt'):
                if option in options:
                    value += '\n' + option + ': ' + inline(options[option])
            extra = {}
            if name == 'corresponding-type-slot' and store:
                rows, dependency = generated_content(name, store, source_id)
                row = next((r for r in rows if r['name']==arg.strip()), None)
                if row:
                    value = f"Corresponding slot ID {arg}: part of the Stable ABI since {row['added']}. {row['ifdef_note']}".strip()
                    extra['derived_from'] = [dependency]
            emit("directive", value, i, i+1, directive_name=name, directive_argument=arg,
                 directive_options=options, **extra)
            # Omit directive options, but do not consume their semantic body.
            if body > i+1:
                i = body; continue
            i += 1; continue

        if re.match(r"\s*\.\. _", line):
            # Preserve link targets for reference resolution in a later pass.
            warnings.append({"line": line_numbers[i], "type": "reference_target", "value": line.strip(), **origin})
            i += 1; continue
        if line.lstrip().startswith("..") and (line.lstrip() == ".." or line.lstrip().startswith(".. ")):
            # Footnotes, citations and substitutions are content, not comments.
            if re.match(r"\s*\.\. (\[|\|)", line):
                emit("reference", inline(line.lstrip()[3:]), i, i+1)
                i += 1; continue
            indent = len(line)-len(line.lstrip()); j = i+1
            while j < len(lines) and (not lines[j].strip() or len(lines[j])-len(lines[j].lstrip()) > indent):
                j += 1
            i = j; continue

        # Literal blocks after :: retain their exact internal indentation.
        if line.rstrip().endswith("::"):
            emit("text", inline(line.rstrip()[:-1]) if line.strip() != "::" else "", i, i+1)
            j = i+1
            while j < len(lines) and not lines[j].strip():
                j += 1
            indent = len(line)-len(line.lstrip())
            if j < len(lines) and len(lines[j])-len(lines[j].lstrip()) > indent:
                end = j
                while end < len(lines) and (not lines[end].strip() or len(lines[end])-len(lines[end].lstrip()) > indent):
                    end += 1
                emit("code", textwrap.dedent("\n".join(lines[j:end])).rstrip(), j, end)
                i = end; continue
            i += 1; continue
        j = i+1
        while j < len(lines) and lines[j].strip() and not lines[j].lstrip().startswith(".."):
            if lines[j].rstrip().endswith("::") or adornment(lines[j]) or (j+1 < len(lines) and adornment(lines[j+1])):
                break
            j += 1
        content = "\n".join(lines[i:j])
        emit("text", inline(content), i, j)
        i = j

    offset = 0
    for index, block in enumerate(blocks):
        block['block_id'] = f'{document_id}:b{index:05d}'
        block["start_char"] = offset
        offset += len(block["text"])
        block["end_char"] = offset
        offset += 2
    return {"title": title or document_id.rsplit("/",1)[-1],
            "metadata": metadata, "sections": sections, "blocks": blocks,
            "text": "\n\n".join(b["text"] for b in blocks),
            "warnings": warnings, "directive_counts": dict(directives),
            "inclusions": inclusions, "omissions": omissions}


def fill_audit_tables(documents):
    events = {}
    for doc in documents:
        for block in doc['blocks']:
            if block.get('directive_name') == 'audit-event':
                argument = block['directive_argument']
                parts = re.findall(r'''"[^"]*"|'[^']*'|\S+''', argument)
                if parts:
                    name = parts[0]
                    args = parts[1].strip('"\'') if len(parts) > 1 else ''
                    record = events.setdefault(name, {'args': set(), 'sources': {}})
                    record['args'].add(args or '')
                    key = (block['source_id'], block['source_repo_path'], block['source_start_line'])
                    record['sources'][key] = {k: block[k] for k in ('source_id', 'source_repo_path', 'source_commit', 'source_sha256', 'source_url', 'source_start_line', 'source_end_line')}
    for doc in documents:
        for block in doc['blocks']:
            if block.get('generated_by') == 'audit-event-table':
                block['text'] = '\n'.join(f"{name} | Arguments: {' / '.join(sorted(record['args'])) or '(none)'}"
                                          for name, record in sorted(events.items()))
                block['derived_from'] = [source for record in events.values() for source in record['sources'].values()]
                block['generated_rows'] = len(events)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=ROOT/"data/raw/python-official")
    ap.add_argument("--output", type=Path, default=ROOT/"data/processed/python-official-v1.1.1")
    ap.add_argument("--support", type=Path, default=ROOT/"data/support/python-official")
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Output already exists; choose a new --output directory.")
    manifest_bytes = (args.input/"manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    sources = {s["source_id"]: s for s in manifest["sources"]}
    store = SnapshotStore(args.input, args.support)
    documents = []
    for entry in sorted(manifest['files'], key=lambda e:e['path']):
        sid, rel = entry['path'].split('/', 1)
        repo_path = sources[sid]['source_subdirectory'] + '/' + rel
        doc_id = manifest['corpus_id'] + '/' + entry['path']
        doc = parse(store.get(sid, repo_path), doc_id, store=store, source_id=sid, repo_path=repo_path)
        doc.update({'document_id': doc_id, 'source_path': entry['path'],
                    'parser_version': VERSION, **store.provenance(sid, repo_path)})
        doc['retrievable'] = not any(o['reason']=='unavailable_generated_NEWS' for o in doc['omissions'])
        if not doc['retrievable']:
            doc['exclusion_reason'] = 'NEWS is generated at Sphinx build time and absent from pinned archive'
        documents.append(doc)
    fill_audit_tables(documents)
    counts, warning_counts, directive_counts = Counter(), Counter(), Counter()
    omissions = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".normalize-") as tmp:
        stage = Path(tmp)
        with (stage/"documents.jsonl").open("w", encoding="utf-8") as out:
            for doc in documents:
                offset = 0
                for block in doc['blocks']:
                    block['start_char'] = offset
                    offset += len(block['text'])
                    block['end_char'] = offset
                    offset += 2
                doc['text'] = '\n\n'.join(b['text'] for b in doc['blocks'])
                doc["text_sha256"] = digest(doc["text"].encode())
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                counts["documents"] += 1
                counts["empty_documents"] += not bool(doc["text"].strip())
                counts['excluded_documents'] += not doc['retrievable']
                counts["normalized_text_bytes"] += len(doc["text"].encode())
                counts["sections"] += len(doc["sections"])
                counts["blocks"] += len(doc["blocks"])
                counts["code_blocks"] += sum(b["kind"] == "code" for b in doc["blocks"])
                counts["pep_metadata_documents"] += "PEP" in doc["metadata"]
                counts['resolved_includes'] += sum(e['status']=='resolved' for e in doc['inclusions'])
                counts['failed_includes'] += sum(e['status']!='resolved' for e in doc['inclusions'])
                counts['unresolved_blocks'] += sum(b['kind']=='unresolved' for b in doc['blocks'])
                counts['generated_blocks'] += sum('generated_by' in b for b in doc['blocks'])
                warning_counts.update(w["type"] for w in doc["warnings"])
                directive_counts.update(doc["directive_counts"])
                omissions.update(o['reason'] for o in doc['omissions'])
        counts['source_bytes'] = sum(e['bytes'] for e in manifest['files'])
        dependencies = store.dependencies()
        dependency_bytes = (json.dumps(dependencies, ensure_ascii=False, indent=2)+'\n').encode()
        (stage/'dependencies.json').write_bytes(dependency_bytes)
        counts['supplemental_files_used'] = sum(not d['primary_document'] for d in dependencies)
        report = {"parser_version": VERSION, "parser_sha256": digest(Path(__file__).read_bytes()),
                  'sources_helper_sha256': digest((ROOT/'scripts/corpus_sources.py').read_bytes()),
                  "input_manifest_sha256": digest(manifest_bytes), "counts": dict(counts),
                  "warnings": dict(warning_counts), "directives": dict(directive_counts),
                  'omissions': dict(omissions),
                  'excluded_documents': [d['source_path'] for d in documents if not d['retrievable']],
                  'dependencies_sha256': digest(dependency_bytes),
                  "documents_sha256": digest((stage/"documents.jsonl").read_bytes()),
                  "limitations": ["Conservative source parser, not full Sphinx rendering", "Generated NEWS document explicitly excluded", "Cross-reference targets recorded, not fully resolved", "RST tables and some substitutions remain in source notation", "Section levels inferred by adornment order", "Visual-only HTML/SVG layout omitted; readable text retained where available"]}
        (stage/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        shutil.move(str(stage), str(args.output))
    store.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
