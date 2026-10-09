"""End-to-end loading: CSV/Parquet on disk -> compact arrays the E-step consumes.

Every loader must produce the same arrays (checked against the pandas baseline):
  correct  int8   (N,)   0/1 answers, sorted by (skill, user, order)
  skill    int32  (N,)   skill code (codes follow sorted skill names)
  starts   int64  (G,)   0-based start of each (skill, user) sequence
  lengths  int64  (G,)
Each loader runs in its own process (peak RSS is per process).

  python load_bench.py make-synth 20000000      # writes synth.csv / synth.parquet
  python load_bench.py run <csv-or-parquet> <loader>
  python load_bench.py all <file>               # every loader, one process each
"""
import sys, os, time, json, resource, subprocess, hashlib
import numpy as np

COLS = ["order_id", "user_id", "skill_name", "correct"]


def finish(skill_codes, user, order, correct):
    """Common tail: stable sort by (skill, user, order), then sequence boundaries."""
    idx = np.lexsort((order, user, skill_codes))
    s, u = skill_codes[idx], user[idx]
    new = np.ones(len(idx), bool)
    new[1:] = (s[1:] != s[:-1]) | (u[1:] != u[:-1])
    starts = np.flatnonzero(new)
    lengths = np.diff(np.append(starts, len(idx)))
    return dict(correct=correct[idx].astype(np.int8), skill=s.astype(np.int32), starts=starts, lengths=lengths)


def codes_sorted(values):
    """Factorize to codes whose order follows sorted unique values (so all loaders agree)."""
    import pandas as pd
    codes, uniq = pd.factorize(values, sort=True)
    return codes


# ---------------- loaders ----------------

def pandas_pybkt_style(path):
    """What pyBKT does today: read every column (latin, low_memory=False), then convert_data (phase 1 version)."""
    import pandas as pd
    from pyBKT.util import data_helper
    df = pd.read_csv(path, low_memory=False, encoding="latin") if path.endswith(".csv") else pd.read_parquet(path)
    if "original" in df.columns:
        df = df[df["original"] == 1]
    df = df.dropna(subset=["skill_name"])
    datas = data_helper.convert_data(df, ".*")
    # not array-identical to the others (pyBKT orders skills by first appearance); report sizes only
    n = sum(d["data"].shape[1] for d in datas.values())
    return None, n


def pandas_usecols(path):
    import pandas as pd
    df = pd.read_csv(path, usecols=COLS + (["original"] if has_original(path) else []), encoding="latin",
                     dtype={"correct": "int8"}, engine="c")
    return from_pandas(df)


def pandas_pyarrow_engine(path):
    import pandas as pd
    df = pd.read_csv(path, usecols=COLS + (["original"] if has_original(path) else []), encoding="latin", engine="pyarrow")
    return from_pandas(df)


def from_pandas(df):
    if "original" in df.columns:
        df = df[df["original"] == 1]
    df = df[df["skill_name"].notna()]
    return finish(codes_sorted(df["skill_name"].to_numpy()), df["user_id"].to_numpy(np.int64),
                  df["order_id"].to_numpy(np.int64), df["correct"].to_numpy(np.int8)), len(df)


def pyarrow_csv(path):
    import pyarrow.csv as pv, pyarrow.compute as pc, pyarrow as pa
    cols = COLS + (["original"] if has_original(path) else [])
    t = pv.read_csv(path, read_options=pv.ReadOptions(encoding="latin1"),
                    convert_options=pv.ConvertOptions(include_columns=cols, strings_can_be_null=True,
                                                      column_types={"skill_name": pa.dictionary(pa.int32(), pa.string())}))
    if "original" in t.column_names:
        t = t.filter(pc.equal(t["original"], 1))
    t = t.filter(pc.is_valid(t["skill_name"]))
    sk = t["skill_name"].combine_chunks()
    # re-code dictionary so codes follow sorted names
    names = np.array(sk.dictionary.to_pylist(), dtype=object)
    rank = np.empty(len(names), np.int32); rank[np.argsort(names, kind="stable")] = np.arange(len(names))
    return finish(rank[sk.indices.to_numpy(zero_copy_only=False)], t["user_id"].to_numpy(), t["order_id"].to_numpy(),
                  t["correct"].to_numpy().astype(np.int8)), t.num_rows


def polars_eager(path):
    import polars as pl
    cols = COLS + (["original"] if has_original(path) else [])
    df = pl.read_csv(path, columns=cols, encoding="utf8-lossy", schema_overrides={"skill_name": pl.Utf8}) if path.endswith(".csv") \
        else pl.read_parquet(path, columns=cols)
    return from_polars(df)


def polars_lazy_sorted(path):
    """Let Polars (Rust, multithreaded) do filter + rank-coding + sort; only arrays come back."""
    import polars as pl
    cols = COLS + (["original"] if has_original(path) else [])
    lf = pl.scan_csv(path, encoding="utf8-lossy") if path.endswith(".csv") else pl.scan_parquet(path)
    lf = lf.select(cols)
    if "original" in cols:
        lf = lf.filter(pl.col("original") == 1)
    lf = (lf.filter(pl.col("skill_name").is_not_null())
            .with_columns(pl.col("skill_name").rank("dense").cast(pl.Int32).alias("skill") - 1)
            .sort(["skill", "user_id", "order_id"], maintain_order=True)
            .select(["skill", "user_id", "correct"]))
    df = lf.collect()
    s, u = df["skill"].to_numpy(), df["user_id"].to_numpy()
    new = np.ones(len(s), bool); new[1:] = (s[1:] != s[:-1]) | (u[1:] != u[:-1])
    starts = np.flatnonzero(new)
    return dict(correct=df["correct"].to_numpy().astype(np.int8), skill=s.astype(np.int32), starts=starts,
                lengths=np.diff(np.append(starts, len(s)))), len(s)


def from_polars(df):
    import polars as pl
    if "original" in df.columns:
        df = df.filter(pl.col("original") == 1)
    df = df.filter(pl.col("skill_name").is_not_null())
    return finish(codes_sorted(df["skill_name"].to_numpy()), df["user_id"].to_numpy(), df["order_id"].to_numpy(),
                  df["correct"].to_numpy().astype(np.int8)), len(df)


def duckdb_sql(path):
    import duckdb
    cols = COLS + (["original"] if has_original(path) else [])
    src = f"read_csv('{path}', ignore_errors=false)" if path.endswith(".csv") else f"read_parquet('{path}')"
    where = "original = 1 AND " if "original" in cols else ""
    q = f"""SELECT (dense_rank() OVER (ORDER BY skill_name) - 1)::INTEGER AS skill, user_id, correct::TINYINT AS correct
            FROM {src} WHERE {where} skill_name IS NOT NULL ORDER BY skill, user_id, order_id"""
    con = duckdb.connect()
    r = con.execute(q).fetchnumpy()
    s, u = r["skill"], r["user_id"]
    new = np.ones(len(s), bool); new[1:] = (s[1:] != s[:-1]) | (u[1:] != u[:-1])
    starts = np.flatnonzero(new)
    return dict(correct=np.asarray(r["correct"], np.int8), skill=np.asarray(s, np.int32), starts=starts,
                lengths=np.diff(np.append(starts, len(s)))), len(s)


def compact_npy(path):
    """Second and later runs: arrays already saved as .npy, memory-mapped (no parse at all)."""
    base = path + ".compact"
    out = {k: np.load(f"{base}.{k}.npy", mmap_mode="r") for k in ("correct", "skill", "starts", "lengths")}
    # touch the data so the timing includes reading it from the page cache
    _ = int(out["correct"].sum()) + int(out["lengths"].sum())
    return out, len(out["correct"])


LOADERS = dict(pandas_pybkt_style=pandas_pybkt_style, pandas_usecols=pandas_usecols, pandas_pyarrow_engine=pandas_pyarrow_engine,
               pyarrow_csv=pyarrow_csv, polars_eager=polars_eager, polars_lazy_sorted=polars_lazy_sorted, duckdb_sql=duckdb_sql,
               compact_npy=compact_npy)


def has_original(path):
    import pyarrow.parquet as pq
    if path.endswith(".parquet"):
        return "original" in pq.read_schema(path).names
    with open(path, encoding="latin1") as f:
        return "original" in f.readline().strip().split(",")


def digest(out):
    h = hashlib.sha1()
    for k in ("correct", "skill", "starts", "lengths"):
        h.update(np.ascontiguousarray(out[k]).astype(np.int64).tobytes())
    return h.hexdigest()[:12]


def run(path, name):
    t = time.perf_counter()
    out, n = LOADERS[name](path)
    secs = time.perf_counter() - t
    res = dict(file=os.path.basename(path), loader=name, rows=n, seconds=round(secs, 3),
               peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), digest=digest(out) if out else "-")
    if out is not None and name == "pandas_usecols" and not os.path.exists(path + ".compact.correct.npy"):
        for k, v in out.items():
            np.save(f"{path}.compact.{k}.npy", v)
    print(json.dumps(res))


def make_synth(rows):
    """Wide-ish CSV like a tutor log: 8 columns, string user and skill ids."""
    import pandas as pd
    rng = np.random.default_rng(0)
    users = rng.integers(0, rows // 25, rows)
    df = pd.DataFrame({"order_id": rng.permutation(rows), "user_id": users, "skill_name": np.char.add("skill_", rng.integers(0, 500, rows).astype(str)),
                       "correct": (rng.random(rows) < 0.65).astype(np.int8), "template_id": rng.integers(0, 5000, rows),
                       "ms_first_response": rng.integers(0, 60000, rows), "answer_text": np.where(rng.random(rows) < .5, "foo", "bar baz"),
                       "hint_count": rng.integers(0, 4, rows)})
    df.to_csv("/tmp/claude-0/data/synth.csv", index=False)
    df.to_parquet("/tmp/claude-0/data/synth.parquet", index=False)
    print("wrote", rows, "rows;", os.path.getsize("/tmp/claude-0/data/synth.csv") >> 20, "MB csv,", os.path.getsize("/tmp/claude-0/data/synth.parquet") >> 20, "MB parquet")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "make-synth":
        make_synth(int(sys.argv[2]))
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3])
    elif cmd == "all":
        names = sys.argv[3].split(",") if len(sys.argv) > 3 else list(LOADERS)
        for name in names:
            r = subprocess.run([sys.executable, __file__, "run", sys.argv[2], name], capture_output=True, text=True)
            print(r.stdout.strip() or f'{{"loader": "{name}", "error": {json.dumps(r.stderr.strip().splitlines()[-1] if r.stderr else "?")}}}')
