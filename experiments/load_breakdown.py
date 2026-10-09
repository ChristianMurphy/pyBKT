"""Split loading into parse vs group/sort for pyarrow and DuckDB on the same file (fresh process each)."""
import sys, time, json, resource, numpy as np
path, which = sys.argv[1], sys.argv[2]
cols = ["order_id", "user_id", "skill_name", "correct"]
t0 = time.perf_counter()
if which == "pyarrow_parse":
    import pyarrow.csv as pv, pyarrow as pa
    t = pv.read_csv(path, convert_options=pv.ConvertOptions(include_columns=cols, column_types={"skill_name": pa.dictionary(pa.int32(), pa.string())}))
    n = t.num_rows
elif which == "pyarrow_parse_sort_arrow":   # sort inside Arrow (Acero), not NumPy
    import pyarrow.csv as pv, pyarrow as pa, pyarrow.compute as pc
    t = pv.read_csv(path, convert_options=pv.ConvertOptions(include_columns=cols))
    t1 = time.perf_counter()
    t = t.sort_by([("skill_name", "ascending"), ("user_id", "ascending"), ("order_id", "ascending")])
    n = t.num_rows; extra = time.perf_counter() - t1
elif which == "pyarrow_parse_sort_numpy":   # what load_bench.py did
    import pyarrow.csv as pv, pyarrow as pa
    t = pv.read_csv(path, convert_options=pv.ConvertOptions(include_columns=cols, column_types={"skill_name": pa.dictionary(pa.int32(), pa.string())}))
    t1 = time.perf_counter()
    sk = t["skill_name"].combine_chunks(); names = np.array(sk.dictionary.to_pylist(), dtype=object)
    rank = np.empty(len(names), np.int32); rank[np.argsort(names, kind="stable")] = np.arange(len(names))
    idx = np.lexsort((t["order_id"].to_numpy(), t["user_id"].to_numpy(), rank[sk.indices.to_numpy(zero_copy_only=False)]))
    n = len(idx); extra = time.perf_counter() - t1
elif which == "duckdb_parse":
    import duckdb
    n = duckdb.connect().execute(f"CREATE TABLE t AS SELECT order_id, user_id, skill_name, correct FROM read_csv('{path}'); SELECT count(*) FROM t").fetchone()[0]
elif which == "duckdb_parse_sort":
    import duckdb
    con = duckdb.connect()
    con.execute(f"CREATE TABLE t AS SELECT order_id, user_id, skill_name, correct FROM read_csv('{path}')")
    t1 = time.perf_counter()
    r = con.execute("SELECT (dense_rank() OVER (ORDER BY skill_name) - 1)::INTEGER AS skill, user_id, correct::TINYINT AS correct FROM t ORDER BY skill, user_id, order_id").fetchnumpy()
    n = len(r["skill"]); extra = time.perf_counter() - t1
el = time.perf_counter() - t0
out = dict(step=which, rows=n, total_s=round(el, 2), peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
if "extra" in dir(): out["sort_part_s"] = round(extra, 2)
print(json.dumps(out))
