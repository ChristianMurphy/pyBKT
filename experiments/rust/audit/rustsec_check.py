"""Match every package in the Cargo.lock files against the local RustSec advisory-db clone.
A version is affected unless it matches a `patched` or `unaffected` requirement (cargo-audit semantics).
Withdrawn advisories are skipped; informational ones (unmaintained/unsound/notice) are listed separately."""
import os, re, sys, tomllib, glob
DB = '/tmp/claude-0/exp/depaudit/advisory-db/crates'
def ver(s):
    s = s.split('+')[0].split('-')[0]
    p = [int(x) for x in s.split('.')] + [0, 0, 0]
    return tuple(p[:3])
def req_ok(v, req):
    for part in req.split(','):
        part = part.strip()
        m = re.match(r'(>=|<=|>|<|=|\^|~)?\s*(.+)', part)
        op, rv = m.group(1) or '^', m.group(2).strip()
        n = len(rv.split('.'))
        r = ver(rv)
        if op == '>=' and not v >= r: return False
        if op == '>' and not v > r: return False
        if op == '<' and not v < r: return False
        if op == '<=' and not v <= r: return False
        if op == '=' and not v == r: return False
        if op in ('^',):
            if r[0] > 0: hi = (r[0] + 1, 0, 0)
            elif r[1] > 0 or n >= 2 and n > 2 and False: hi = (0, r[1] + 1, 0)
            elif n >= 2 and r[1] > 0: hi = (0, r[1] + 1, 0)
            elif n == 1: hi = (1, 0, 0)
            elif n == 2: hi = (0, r[1] + 1, 0)
            else: hi = (0, 0, r[2] + 1) if r[1] == 0 else (0, r[1] + 1, 0)
            if not (r <= v < hi): return False
        if op == '~':
            hi = (r[0], r[1] + 1, 0) if n >= 2 else (r[0] + 1, 0, 0)
            if not (r <= v < hi): return False
    return True
def packages(lock):
    d = tomllib.load(open(lock, 'rb'))
    return {(p['name'], p['version']) for p in d['package']}
locks = {'bkt_rs': 'bkt_rs/Cargo.lock', 'bkt_lean': 'bkt_lean/Cargo.lock'}
for label, lock in locks.items():
    pk = packages(lock)
    hits, info = [], []
    for name, v in sorted(pk):
        for fn in glob.glob(f'{DB}/{name}/*.md'):
            txt = open(fn).read()
            meta = tomllib.loads(re.search(r'```toml\n(.*?)\n```', txt, re.S).group(1))
            adv = meta['advisory']
            if adv.get('withdrawn'): continue
            vs = meta.get('versions', {})
            reqs = vs.get('patched', []) + vs.get('unaffected', [])
            affected = not any(req_ok(ver(v), r) for r in reqs)
            if affected:
                (info if adv.get('informational') else hits).append(f"{name} {v}: {adv['id']} ({adv.get('informational', 'vulnerability')}) {re.search(r'^# (.*)$', txt, re.M).group(1)}")
    print(f'{label}: {len(pk)} locked packages (incl. build/proc-macro and optional deps); advisories affecting resolved versions: {len(hits)}; informational: {len(info)}')
    for h in hits + info: print('   ', h)
