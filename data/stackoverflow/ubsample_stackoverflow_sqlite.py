#!/usr/bin/env python
import os, sqlite3, random, math, argparse
from contextlib import closing
from collections import defaultdict

def stratify_bucket(n):
    if n <= 5: return "1-5"
    if n <= 20: return "6-20"
    if n <= 100: return "21-100"
    return "100+"

def open_ro(db):
    uri = f"file:{os.path.abspath(db)}?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)

def open_rw_new(db):
    if os.path.exists(db): os.remove(db)
    conn = sqlite3.connect(db, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA mmap_size=268435456;")
    cur.execute("CREATE TABLE examples (split_name TEXT, client_id TEXT, serialized_example_proto BLOB);")
    cur.execute("CREATE INDEX idx_examples_split_client ON examples(split_name, client_id);")
    conn.commit()
    return conn

def choose_k(pop, k):
    if k >= len(pop): return list(pop)
    return random.sample(pop, k)

def main(args):
    random.seed(args.seed)
    with closing(open_ro(args.src)) as src, closing(open_rw_new(args.dst)) as dst:
        s_cur, d_cur = src.cursor(), dst.cursor()

        # 1) 统计每用户样本数
        print("[1/4] counting per-user examples...")
        s_cur.execute("""
            SELECT split_name, client_id, COUNT(*) as n
            FROM examples
            GROUP BY split_name, client_id
        """)
        per_user = s_cur.fetchall()

        # 2) 分层：每 split 内按 bucket 聚合
        print("[2/4] bucketing clients...")
        buckets = {}  # split -> bucket -> list[(client_id, n)]
        #counts_by_split = {"train":0,"val":0,"test":0}
        counts_by_split = defaultdict(int)
        for split, cid, n in per_user:
            b = stratify_bucket(n)
            buckets.setdefault(split, {}).setdefault(b, []).append((cid, n))
            counts_by_split[split] = counts_by_split.get(split,0)+1

        # 3) 计算目标用户数
        tgt = {}
        splits_present = list(counts_by_split.keys())
        if args.fraction is not None:
            for split, tot in counts_by_split.items():
                tgt[split] = max(1, int(round(tot * args.fraction)))
        else:
            if args.target_clients_per_split is None:
                raise SystemExit("Either --fraction or --target_clients_per_split is required.")
            for split in splits_present:
                tgt[split] = args.target_clients_per_split
        # tgt = {}
        # if args.fraction is not None:
        #     for split, tot in counts_by_split.items():
        #         tgt[split] = max(1, int(round(tot * args.fraction)))
        # else:
        #     if args.target_clients_per_split is None:
        #         raise SystemExit("Either --fraction or --target_clients_per_split is required.")
        #     for split in ("train","val","test"):
        #         tgt[split] = args.target_clients_per_split

        # 4) 在每个 split 内，按 bucket 占比分配 quota 并抽样
        print("[3/4] sampling clients...")
        #chosen = { "train": [], "val": [], "test": [] }
        chosen = {s: [] for s in splits_present}
        for split, bmap in buckets.items():
            # 该 split 的总用户与各 bucket 占比
            tot = sum(len(v) for v in bmap.values())
            # 先按占比分配，四舍五入；再根据误差做微调
            alloc = {}
            for b, lst in bmap.items():
                alloc[b] = int(round(len(lst) / tot * tgt.get(split, 0)))
            # 微调到刚好等于目标
            diff = tgt.get(split,0) - sum(alloc.values())
            # 正 diff：补齐；负 diff：削减
            keys = list(bmap.keys())
            i = 0
            while diff != 0 and keys:
                k = keys[i % len(keys)]
                if diff > 0:
                    alloc[k] += 1; diff -= 1
                else:
                    if alloc[k] > 0: alloc[k] -= 1; diff += 1
                i += 1
            # 真正抽样
            for b, lst in bmap.items():
                k = alloc[b]
                if k <= 0: continue
                chosen[split].extend(choose_k(lst, k))

        # 5) 把抽中的 client 的行复制到新库（可选：限制每用户最多样本数）
        print("[4/4] copying rows to dst sqlite...")
        copied_rows = 0
        for split in splits_present:
            sel = set(cid for cid, _ in chosen.get(split, []))
            if not sel: continue
            # 分批执行（placeholders 太多容易慢）
            batch = 0
            for cid in sel:
                if args.max_examples_per_client and args.max_examples_per_client > 0:
                    # 先取所有行，再截断到上限（随机）
                    rows = s_cur.execute(
                        "SELECT serialized_example_proto FROM examples WHERE split_name=? AND client_id=?",
                        (split, cid)
                    ).fetchall()
                    if len(rows) > args.max_examples_per_client:
                        rows = choose_k(rows, args.max_examples_per_client)
                    d_cur.executemany(
                        "INSERT INTO examples(split_name, client_id, serialized_example_proto) VALUES(?,?,?)",
                        [(split, cid, r[0]) for r in rows]
                    )
                    copied_rows += len(rows)
                else:
                    # 直接整 client 复制（最快）
                    d_cur.execute("""
                        INSERT INTO examples(split_name, client_id, serialized_example_proto)
                        SELECT split_name, client_id, serialized_example_proto
                        FROM examples WHERE split_name=? AND client_id=?""", (split, cid))
                    copied_rows += d_cur.rowcount if d_cur.rowcount != -1 else 0
                batch += 1
                if batch % 200 == 0:
                    dst.commit()
            dst.commit()
        print(f"done. copied rows ~ {copied_rows:,}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原 60GB sqlite 路径")
    ap.add_argument("--dst", required=True, help="输出精简 sqlite 路径")
    ap.add_argument("--fraction", type=float, default=0.1, help="每个 split 抽样比例（0-1），与 target_clients_per_split 二选一")
    ap.add_argument("--target_clients_per_split", type=int, default=None, help="每个 split 目标客户端数，优先于 fraction")
    ap.add_argument("--max_examples_per_client", type=int, default=0, help="每客户端最多保留样本数（0 表示不限制）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
