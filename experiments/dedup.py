"""How much exact E-step work can sharing be saved? Per skill on ASSISTments (and synthetic):
attempts vs distinct whole sequences vs distinct prefixes (trie nodes). With forward-only smoothing
(phi, rho per trie node, stats read at sequence ends weighted by count), the exact E-step costs one
update per trie node instead of one per attempt."""
import numpy as np, pandas as pd
from bkt_np import *

def trie_stats(seqs, key=lambda s: tuple(s)):
    nodes = set(); whole = set()
    for s in seqs:
        t = key(s); whole.add(t)
        for i in range(1, len(t) + 1):
            nodes.add(t[:i])
    return len(whole), len(nodes)

df = pd.read_csv("/tmp/claude-0/data/as.csv", encoding="latin", low_memory=False)
df = df[df["original"] == 1].dropna(subset=["skill_name"]).sort_values("order_id")
rows = []
for skill, d in df.groupby("skill_name"):
    seqs = [g.to_numpy() for _, g in d.groupby("user_id", sort=False)["correct"]]
    tseqs = [tuple(zip(g["template_id"], g["correct"])) for _, g in d.groupby("user_id", sort=False)]
    w, n = trie_stats(seqs)
    _, nt = trie_stats(tseqs, key=lambda s: s)
    rows.append(dict(skill=skill, attempts=len(d), students=len(seqs), distinct_seqs=w, trie_nodes=n, trie_nodes_with_template=nt))
o = pd.DataFrame(rows)
tot = o[["attempts", "students", "distinct_seqs", "trie_nodes", "trie_nodes_with_template"]].sum()
print("ASSISTments, all", len(o), "skills:", tot.to_dict())
print("work ratio attempts/trie_nodes:", round(tot.attempts / tot.trie_nodes, 2), "| with template in key:", round(tot.attempts / tot.trie_nodes_with_template, 2))
big = o.sort_values("attempts", ascending=False).head(5)
big["ratio"] = (big.attempts / big.trie_nodes).round(2)
print(big.to_string(index=False))
# synthetic: 250k students x 20 attempts from a BKT model (binary, single template)
rng = np.random.default_rng(0)
S, L = 250_000, 20
known = rng.random(S) < .3; cols = []
for t in range(L):
    r = rng.random(S); cols.append(np.where(known, r >= .1, r < .2).astype(np.int8)); known |= rng.random(S) < .15
M = np.stack(cols, 1)
# trie node count = number of distinct prefixes at each depth
nodes = sum(len(np.unique(np.packbits(M[:, :k], axis=1, bitorder="little").view(np.uint8).reshape(S, -1), axis=0)) for k in range(1, L + 1))
print("synthetic 5M attempts: trie nodes", nodes, "ratio", round(S * L / nodes, 2))
