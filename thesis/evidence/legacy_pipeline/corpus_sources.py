"""Verified, read-only access to pinned repository files, without executing them."""
from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def resolve_path(current, target, docroot):
    if "://" in target or "\\" in target or not target.strip():
        raise ValueError(f"Invalid repository include path: {target!r}")
    path = posixpath.normpath(posixpath.join(docroot, target.lstrip('/'))
                             if target.startswith('/') else
                             posixpath.join(posixpath.dirname(current), target))
    if path == '..' or path.startswith('../') or path.startswith('/'):
        raise ValueError(f"Include escapes repository: {target}")
    return path


def prepare(manifest_path, output, cache=None):
    manifest = json.loads(manifest_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    for source in manifest['sources']:
        dest = output / f"{source['source_id']}-{source['commit']}.tar.gz"
        expected = source['archive_sha256']
        if dest.exists():
            if sha(dest.read_bytes()) != expected:
                raise ValueError(f"Existing archive checksum mismatch: {dest}")
            continue
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            staged = Path(tmp)/'download.tar.gz'
            local = cache / ('cpython.tar.gz' if source['source_id']=='cpython-docs' else 'peps.tar.gz') if cache else None
            if local and local.exists():
                shutil.copyfile(local, staged)
            else:
                request = urllib.request.Request(source['archive_url'], headers={'User-Agent': 'rag-thesis/1.1'})
                with urllib.request.urlopen(request, timeout=180) as response, staged.open('wb') as stream:
                    shutil.copyfileobj(response, stream)
            if sha(staged.read_bytes()) != expected:
                raise ValueError(f"Archive checksum mismatch: {source['source_id']}")
            staged.rename(dest)
        print(f"Verified archive: {dest.name}")


class SnapshotStore:
    def __init__(self, raw, support):
        self.raw = raw
        self.manifest = json.loads((raw/'manifest.json').read_text())
        self.sources = {s['source_id']: s for s in self.manifest['sources']}
        self.files, self.primary, self.archives, self.members, self.used = {}, {}, {}, {}, {}
        for e in self.manifest['files']:
            sid, rel = e['path'].split('/', 1)
            repo_path = self.sources[sid]['source_subdirectory'] + '/' + rel
            data = (raw/e['path']).read_bytes()
            if sha(data) != e['sha256'] or len(data) != e['bytes']:
                raise ValueError(f"Source checksum mismatch: {e['path']}")
            self.files[sid, repo_path] = data.decode('utf-8')
            self.primary[sid, repo_path] = e['path']
        for sid, s in self.sources.items():
            path = support / f"{sid}-{s['commit']}.tar.gz"
            if sha(path.read_bytes()) != s['archive_sha256']:
                raise ValueError(f"Archive checksum mismatch: {sid}")
            archive = tarfile.open(path, 'r:gz')
            self.archives[sid] = archive
            self.members[sid] = {m.name.split('/',1)[1]: m for m in archive.getmembers()
                                 if m.isfile() and '/' in m.name}

    def close(self):
        for archive in self.archives.values():
            archive.close()

    def get(self, sid, path):
        key = (sid, path)
        if key not in self.files:
            member = self.members[sid].get(path)
            if member is None:
                raise FileNotFoundError(f"Not in pinned repository: {sid}/{path}")
            with self.archives[sid].extractfile(member) as stream:
                self.files[key] = stream.read().decode('utf-8')
        return self.files[key]

    def provenance(self, sid, path):
        text = self.get(sid, path)
        source = self.sources[sid]
        record = {'source_id': sid, 'source_repo_path': path,
                  'source_commit': source['commit'], 'source_sha256': sha(text.encode('utf-8')),
                  'source_url': f"{source['repository']}/blob/{source['commit']}/{path}"}
        self.used[sid, path] = {**record, 'bytes': len(text.encode('utf-8')),
                               'primary_document': (sid, path) in self.primary}
        return record

    def resolve(self, sid, current, target):
        return resolve_path(current, target, self.sources[sid]['source_subdirectory'])

    def dependencies(self):
        return [self.used[k] for k in sorted(self.used)]


def select_lines(text, options):
    """Implement the literal/include range options; return lines + original IDs."""
    rows = list(enumerate(text.splitlines(), start=1))
    if 'start-line' in options or 'end-line' in options:
        rows = rows[int(options.get('start-line', 0)):int(options['end-line']) if 'end-line' in options else None]
    for key in ('start-at', 'start-after', 'end-at', 'end-before'):
        if key not in options:
            continue
        index = next((i for i, (_, line) in enumerate(rows) if options[key] in line), None)
        if index is None:
            raise ValueError(f"Missing {key} marker: {options[key]!r}")
        rows = rows[index + (key=='start-after'):] if key.startswith('start') else rows[:index + (key=='end-at')]
    if 'lines' in options:
        selected = set()
        for item in options['lines'].split(','):
            ends = item.strip().split('-')
            if len(ends) == 1:
                selected.add(int(ends[0]))
            else:
                selected.update(range(int(ends[0] or 1), int(ends[1] or len(rows))+1))
        if any(n < 1 or n > len(rows) for n in selected):
            raise ValueError('Line range outside included file')
        rows = [row for i, row in enumerate(rows, 1) if i in selected]
    return [line for _, line in rows], [n for n, _ in rows]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/raw/python-official/manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT/'data/support/python-official')
    parser.add_argument('--cache', type=Path)
    args = parser.parse_args()
    prepare(args.manifest, args.output, args.cache)
