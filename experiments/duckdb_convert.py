"""convert_data in one DuckDB query: CSV/Parquet -> compact arrays for every skill at once.

What runs in SQL:
  * filter original == 1 and missing skills; the 0/1 -> 1/2 answer code
  * order by (skill, user, order_id) with a deterministic tie-break (file row number). pyBKT's
    quicksort leaves ties in arbitrary order; see experiments README sec. 8
  * skill codes, per-student starts and lengths (window functions)
  * multigs / multilearn codes: templates ranked within each skill, as pyBKT does (sorted unique values)
  * multipair: LAG(problem) within each student, pairs numbered by first appearance within the skill
    (pyBKT's numbering; its key strings are numpy reprs and are not reproduced, see sec. 8)
Python only receives the final columns (fetchnumpy) and slices them per skill.

  python duckdb_convert.py check <csv>        # compare with pyBKT convert_data (students without tied order_ids)
  python duckdb_convert.py time  <csv|parquet> # time + peak memory for a full multi-model conversion
"""
import sys, time, json, resource
import numpy as np
import duckdb

SQL = """
WITH src AS (
    SELECT row_number() OVER () AS rid, * FROM {reader}
), base AS (
    SELECT rid, CAST(skill_name AS VARCHAR) AS skill_name, user_id, order_id, correct, template_id, problem_id
    FROM src WHERE {where} skill_name IS NOT NULL
), ordered AS (
    SELECT *, row_number() OVER (ORDER BY skill_name, user_id, order_id, rid) - 1 AS pos,
           LAG(problem_id) OVER (PARTITION BY skill_name, user_id ORDER BY order_id, rid) AS prev_problem
    FROM base
), pairs AS (   -- first appearance of each (problem, previous problem) pair inside its skill
    SELECT skill_name, problem_id, prev_problem,
           row_number() OVER (PARTITION BY skill_name ORDER BY min(pos)) + 1 AS pair_resource   -- 1 = "Default"
    FROM ordered WHERE prev_problem IS NOT NULL GROUP BY skill_name, problem_id, prev_problem
)
SELECT (dense_rank() OVER (ORDER BY o.skill_name) - 1)::INTEGER AS skill,
       o.user_id,
       (o.correct + 1)::TINYINT AS answer,                                                  -- pyBKT codes: 1 wrong, 2 right
       (dense_rank() OVER (PARTITION BY o.skill_name ORDER BY o.template_id) - 1)::INTEGER AS gs_code,   -- multigs (0-based)
       dense_rank() OVER (PARTITION BY o.skill_name ORDER BY o.template_id)::INTEGER AS learn_resource,   -- multilearn (1-based)
       coalesce(p.pair_resource, 1)::INTEGER AS pair_resource                               -- multipair
FROM ordered o LEFT JOIN pairs p
  ON p.skill_name = o.skill_name AND p.problem_id = o.problem_id AND p.prev_problem = o.prev_problem
ORDER BY o.pos
"""


def reader(path):
    return f"read_parquet('{path}')" if path.endswith(".parquet") else f"read_csv('{path}', ignore_errors=false)"


def build_sql(path, cols, model):
    where = "original = 1 AND" if "original" in cols else ""
    pair_col = "problem_id" if "problem_id" in cols else "template_id"
    if model == "multipair":
        return SQL.format(reader=reader(path), where=where).replace("template_id, problem_id", f"template_id, {pair_col} AS problem_id")
    extra = {"default": "", "multigs": ", (dense_rank() OVER (PARTITION BY skill_name ORDER BY template_id) - 1)::INTEGER AS gs_code",
             "multilearn": ", dense_rank() OVER (PARTITION BY skill_name ORDER BY template_id)::INTEGER AS learn_resource"}[model]
    return f"""
        SELECT (dense_rank() OVER (ORDER BY skill_name) - 1)::INTEGER AS skill, user_id, (correct + 1)::TINYINT AS answer {extra}
        FROM (SELECT row_number() OVER () AS rid, * FROM {reader(path)}) WHERE {where} skill_name IS NOT NULL
        ORDER BY skill, user_id, order_id, rid"""


def convert(path, threads=None, model="all"):
    con = duckdb.connect()
    if threads:
        con.execute(f"SET threads={threads}")
    cols = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM {reader(path)}").fetchall()]
    where = "original = 1 AND" if "original" in cols else ""
    if model == "all":
        sql = SQL.format(reader=reader(path), where=where)
        if "problem_id" not in cols:   # synthetic file: pair on template_id instead
            sql = sql.replace("template_id, problem_id", "template_id, template_id AS problem_id")
    else:
        sql = build_sql(path, cols, model)
    r = con.execute(sql).fetchnumpy()
    out = {k: np.asarray(v) for k, v in r.items()}
    # student boundaries (sorted by skill, user): cheap vectorised pass on the returned columns
    s, u = out["skill"], out["user_id"]
    new = np.ones(len(s), bool)
    new[1:] = (s[1:] != s[:-1]) | (u[1:] != u[:-1])
    out["starts"] = np.flatnonzero(new)
    out["lengths"] = np.diff(np.append(out["starts"], len(s)))
    out["student_skill"] = s[out["starts"]]
    return out


def check(path):
    import pandas as pd
    from pyBKT.util import data_helper
    out = convert(path)
    df = pd.read_csv(path, low_memory=False, encoding="latin")
    df = df[df["original"] == 1].dropna(subset=["skill_name"])
    tied = df.duplicated(["user_id", "skill_name", "order_id"], keep=False)
    tied_users = set(map(tuple, df.loc[tied, ["skill_name", "user_id"]].astype(str).values))
    names = sorted(df.skill_name.astype(str).unique())
    results = {}
    for mt, label in [(None, "default"), ([0, 0, 0, 1], "multigs"), ([1, 0, 0, 0], "multilearn"), ([0, 0, 1, 0], "multipair")]:
        ref = data_helper.convert_data(df.copy(), ".*", model_type=mt, defaults={"multipair": "problem_id"})
        checked = mismatched = 0
        for code, name in enumerate(names):
            if name not in ref:
                continue    # e.g. the regex-character skill pyBKT cannot match (README sec. 8)
            d = ref[name]
            sel = np.flatnonzero(out["student_skill"] == code)
            for j, (st, ln) in enumerate(zip(d["starts"] - 1, d["lengths"])):
                k = sel[j]
                if (name, str(out["user_id"][out["starts"][k]])) in tied_users:
                    continue
                a, b = out["starts"][k], out["starts"][k] + out["lengths"][k]
                if label == "multigs":
                    mine = out["gs_code"][a:b]; theirs = d["data"][:, st:st + ln].argmax(axis=0)
                elif label == "multilearn":
                    mine = out["learn_resource"][a:b]; theirs = d["resources"][st:st + ln]
                elif label == "multipair":
                    mine = out["pair_resource"][a:b]; theirs = d["resources"][st:st + ln]
                else:
                    mine = out["answer"][a:b]; theirs = d["data"][0, st:st + ln]
                checked += 1
                mismatched += int(ln != out["lengths"][k] or not np.array_equal(mine, theirs))
        results[label] = dict(students_checked=checked, mismatched=mismatched)
    print(json.dumps(results, indent=1))


def timed(path, model="all"):
    t = time.perf_counter()
    out = convert(path, model=model)
    print(json.dumps(dict(file=path.split("/")[-1], model=model, rows=len(out["answer"]), students=len(out["starts"]), skills=int(out["skill"].max()) + 1,
                          seconds=round(time.perf_counter() - t, 2), peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                          bytes_per_row_out=round(sum(v.nbytes for v in out.values()) / len(out["answer"]), 1))))


if __name__ == "__main__":
    {"check": check, "time": timed}[sys.argv[1]](*sys.argv[2:])
