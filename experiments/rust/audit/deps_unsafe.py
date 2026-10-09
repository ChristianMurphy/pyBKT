"""Dependency count + `unsafe` line count (comments excluded) for each crate's resolved tree.
usage: python deps_unsafe.py  (run from /tmp/claude-0/exp/rust)"""
import subprocess, re, os, json, sys
REG = '/root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f'
ENV = dict(os.environ, PATH='/root/.cargo/bin:' + os.environ['PATH'])

def tree(crate_dir, feats, edges):
    cmd = ['cargo', 'tree', '-e', edges, '--prefix', 'none', '--format', '{p}|{f}'] + feats
    out = subprocess.run(cmd, cwd=crate_dir, capture_output=True, text=True, env=ENV, check=True).stdout
    pk = set()
    for line in out.splitlines():
        p = line.split('|')[0].replace(' (*)', '').replace(' (proc-macro)', '').strip()
        if p: pk.add(' '.join(p.split()[:2]))
    return pk

def unsafe_lines(path):
    n = 0
    for dp, dns, fns in os.walk(path):
        dns[:] = [d for d in dns if d not in ('tests', 'benches', 'examples', 'target', 'fuzz')]
        for fn in fns:
            if not fn.endswith('.rs'): continue
            src = open(os.path.join(dp, fn), encoding='utf-8', errors='replace').read()
            src = re.sub(r'/\*.*?\*/', lambda m: '\n' * m.group(0).count('\n'), src, flags=re.S)  # block comments
            for line in src.splitlines():
                code = re.sub(r'//.*', '', line)  # line + doc comments (approx.: ignores '//' inside strings)
                code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', code)
                if re.search(r'\bunsafe\b', code): n += 1
    return n

rows = {}
for label, d, feats in [('bkt_rs', 'bkt_rs', []), ('bkt_lean (default)', 'bkt_lean', []), ('bkt_lean +simd', 'bkt_lean', ['--features', 'simd'])]:
    runtime = tree(d, feats, 'normal,no-proc-macro')
    allp = tree(d, feats, 'normal,build')
    proc = tree(d, feats, 'normal') - set()  # placeholder
    res = []
    for p in sorted(allp):
        name, ver = p.split()
        ver = ver.lstrip('v')
        if name.startswith('bkt_'):
            path = os.path.join(d, 'src')
        else:
            path = os.path.join(REG, f'{name}-{ver}')
        res.append((name, ver, unsafe_lines(path), p in runtime))
    rows[label] = res
    rt = [r for r in res if r[3]]
    print(f'\n### {label}: {len(rt)} crates linked into the extension (normal edges, no proc-macros) incl. itself, {len(res)} incl. build deps / proc-macros')
    print('| crate | version | unsafe lines (non-comment) | linked into the .so? |\n|---|---|---:|---|')
    for name, ver, u, r in res:
        print(f'| {name} | {ver} | {u} | {"yes" if r else "no (build-time / proc-macro)"} |')
    print(f'| **total** | | **{sum(r[2] for r in res)}** (linked: {sum(r[2] for r in res if r[3])}) | |')
json.dump(rows, open('audit/deps_unsafe.json', 'w'), indent=1)
