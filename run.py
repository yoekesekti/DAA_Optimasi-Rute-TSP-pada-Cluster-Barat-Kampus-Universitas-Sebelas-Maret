import argparse
import json
import math
import os
import time
import heapq
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:
    psutil = None


# =========================
# Haversine + Distance Matrix
# =========================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c * 1000.0  # meter


def build_dist_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i, j] = np.inf
            else:
                dist[i, j] = haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
    return dist


# =========================
# Reduce Matrix (BnB)
# =========================
def reduce_matrix(mat):
    m = mat.copy()
    n = m.shape[0]
    reduction_cost = 0.0

    # reduce row
    for i in range(n):
        row = m[i]
        finite = row[np.isfinite(row)]
        if finite.size:
            min_val = finite.min()
            if min_val > 0:
                reduction_cost += float(min_val)
                m[i, np.isfinite(m[i])] -= min_val

    # reduce col
    for j in range(n):
        col = m[:, j]
        finite = col[np.isfinite(col)]
        if finite.size:
            min_val = finite.min()
            if min_val > 0:
                reduction_cost += float(min_val)
                m[np.isfinite(m[:, j]), j] -= min_val

    return m, reduction_cost


# =========================
# Backtracking (MEOS)
# =========================
def tsp_backtracking(dist, start=0, time_limit=None, max_nodes_explore=None):
    n = dist.shape[0]
    visited = [False] * n
    visited[start] = True

    best_cost = float("inf")
    best_route = None

    nodes_explored = 0
    pruned_count = 0
    iteration_best_found = None
    cutoff = False

    start_time = time.perf_counter()

    process = psutil.Process(os.getpid()) if psutil else None
    peak_memory = 0.0

    # MEOS: minimal outgoing edge
    min_out = np.min(np.where(np.isfinite(dist), dist, np.inf), axis=1)
    min_out[~np.isfinite(min_out)] = 0

    def dfs(current, depth, cost, path):
        nonlocal best_cost, best_route, nodes_explored, pruned_count
        nonlocal iteration_best_found, cutoff, peak_memory

        if process:
            cur_mem = process.memory_info().rss / (1024 * 1024)
            peak_memory = max(peak_memory, cur_mem)

        if time_limit and (time.perf_counter() - start_time) > time_limit:
            cutoff = True
            return
        if max_nodes_explore and nodes_explored >= max_nodes_explore:
            cutoff = True
            return

        nodes_explored += 1

        unvisited = np.where(~np.array(visited))[0]
        est_lower = cost + float(np.sum(min_out[unvisited]))
        if est_lower >= best_cost:
            pruned_count += 1
            return

        if depth == n:
            total_cost = cost + dist[current, start]
            if total_cost < best_cost:
                best_cost = total_cost
                best_route = path + [start]
                iteration_best_found = nodes_explored
            return

        for nxt in range(n):
            if (not visited[nxt]) and np.isfinite(dist[current, nxt]):
                visited[nxt] = True
                dfs(nxt, depth + 1, cost + dist[current, nxt], path + [nxt])
                visited[nxt] = False
                if cutoff:
                    return

    dfs(start, 1, 0.0, [start])

    return {
        "best_cost": float(best_cost),
        "best_route": best_route,
        "nodes_explored": int(nodes_explored),
        "pruned_count": int(pruned_count),
        "iteration_best_found": iteration_best_found,
        "runtime_s": float(time.perf_counter() - start_time),
        "cutoff": bool(cutoff),
        "peak_memory_mb": float(peak_memory),
    }


# =========================
# Branch and Bound (PQ + reduction)
# =========================
def branch_and_bound_tsp(dist, start=0, time_limit=None):
    n = dist.shape[0]

    root = dist.copy()
    np.fill_diagonal(root, np.inf)
    root, root_lb = reduce_matrix(root)

    visited0 = [False] * n
    visited0[start] = True

    heap = []
    heapq.heappush(heap, (root_lb, start, 1, [start], visited0, root, root_lb, 0.0))

    best_cost = float("inf")
    best_route = None

    nodes_expanded = 0
    pruned_count = 0
    generated_children = 0
    iteration_best_found = None
    max_queue_size = 1

    start_time = time.perf_counter()
    cutoff = False

    process = psutil.Process(os.getpid()) if psutil else None
    peak_memory = 0.0

    while heap:
        if process:
            cur_mem = process.memory_info().rss / (1024 * 1024)
            peak_memory = max(peak_memory, cur_mem)

        if time_limit and (time.perf_counter() - start_time) > time_limit:
            cutoff = True
            break

        bound, curr, level, path, visited, red_mat, lb_cost, true_cost = heapq.heappop(heap)
        nodes_expanded += 1
        max_queue_size = max(max_queue_size, len(heap) + 1)

        if bound >= best_cost:
            pruned_count += 1
            continue

        if level == n:
            total_cost = true_cost + dist[curr, start]
            if total_cost < best_cost:
                best_cost = total_cost
                best_route = path + [start]
                iteration_best_found = nodes_expanded
            continue

        for j in range(n):
            if visited[j] or not np.isfinite(dist[curr, j]):
                continue

            generated_children += 1

            child_visited = visited.copy()
            child_visited[j] = True

            child_mat = red_mat.copy()
            child_mat[curr, :] = np.inf
            child_mat[:, j] = np.inf
            child_mat[j, start] = np.inf

            child_mat, red_cost = reduce_matrix(child_mat)
            new_bound = lb_cost + red_mat[curr, j] + red_cost

            if new_bound < best_cost:
                heapq.heappush(
                    heap,
                    (new_bound, j, level + 1, path + [j], child_visited, child_mat, new_bound, true_cost + dist[curr, j]),
                )
            else:
                pruned_count += 1

    return {
        "best_cost": float(best_cost),
        "best_route": best_route,
        "nodes_expanded": int(nodes_expanded),
        "generated_children": int(generated_children),
        "pruned_count": int(pruned_count),
        "iteration_best_found": iteration_best_found,
        "max_queue_size": int(max_queue_size),
        "runtime_s": float(time.perf_counter() - start_time),
        "cutoff": bool(cutoff),
        "peak_memory_mb": float(peak_memory),
    }


# =========================
# Load JSON instance
# =========================
def load_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        inst = json.load(f)

    nodes = sorted(inst["nodes"], key=lambda x: x["id"])
    names = [n.get("name", f"Node_{i}") for i, n in enumerate(nodes)]
    coords = [(float(n["latitude"]), float(n["longitude"])) for n in nodes]
    start_index = int(inst.get("start_index", 0))
    return inst.get("instance_id", os.path.basename(path)), names, coords, start_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", help="Folder berisi instance JSON")
    ap.add_argument("--instance", default="", help="Kalau diisi, run 1 file saja (mis: tsp_G01.json)")
    ap.add_argument("--sizes", default="8,10,12,15", help="Ukuran n yang diuji (comma separated)")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--base_seed", type=int, default=12345)
    ap.add_argument("--time_limit", type=float, default=60.0)
    ap.add_argument("--max_backtracking_n", type=int, default=10)
    ap.add_argument("--bt_node_limit", type=int, default=10_000_000)
    ap.add_argument("--out_dir", default="results", help="Folder output results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sizes_to_test = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    experiment_id = datetime.now().strftime("exp_%Y%m%d_%H%M%S")

    # pilih file
    if args.instance:
        files = [os.path.join(args.data_dir, args.instance)]
    else:
        files = [
            os.path.join(args.data_dir, f)
            for f in sorted(os.listdir(args.data_dir))
            if f.lower().endswith(".json")
        ]

    if not files:
        raise RuntimeError("Tidak ada file .json di data_dir.")

    results = []

    for path in files:
        instance_id, names_all, coords_all, start = load_instance(path)
        N_ALL = len(coords_all)

        dist_all = build_dist_matrix(coords_all)

        print("\n=====================================")
        print(f"INSTANCE: {instance_id} | total_nodes={N_ALL} | start={start}")
        print("=====================================")

        for sz in sizes_to_test:
            if sz > N_ALL:
                print(f"[SKIP] n={sz} > total_nodes={N_ALL}")
                continue

            # pilih subset node (ngikutin notebook: random choice)
            rng = np.random.default_rng(args.base_seed + sz)
            indices = rng.choice(np.arange(N_ALL), size=sz, replace=False).tolist()

            sub_names = [names_all[i] for i in indices]
            sub_dist = dist_all[np.ix_(indices, indices)]

            print(f"\n--- Eksperimen n={sz} (repeats={args.repeats}) ---")

            for r in range(args.repeats):
                run_seed = args.base_seed + sz * 1000 + r
                np.random.seed(run_seed)

                # Branch & Bound
                bnb = branch_and_bound_tsp(sub_dist, start=0, time_limit=args.time_limit)
                bnb_route_idx = bnb["best_route"]
                bnb_route_names = [sub_names[i] for i in bnb_route_idx] if bnb_route_idx else None

                results.append({
                    "experiment_id": experiment_id,
                    "instance_id": instance_id,
                    "n": sz,
                    "repeat": r,
                    "seed": run_seed,
                    "algorithm": "branch_and_bound",
                    "best_cost_m": float(bnb["best_cost"]),
                    "runtime_s": float(bnb["runtime_s"]),
                    "cutoff": bool(bnb["cutoff"]),
                    "nodes_expanded": int(bnb["nodes_expanded"]),
                    "generated_children": int(bnb["generated_children"]),
                    "pruned_count": int(bnb["pruned_count"]),
                    "max_queue_size": int(bnb["max_queue_size"]),
                    "iteration_best_found": bnb["iteration_best_found"],
                    "peak_memory_mb": float(bnb["peak_memory_mb"]),
                    "route_idx": bnb_route_idx,
                    "route_names": bnb_route_names,
                    "nodes_used": ";".join(sub_names),
                })

                # Backtracking (hanya kalau n kecil)
                if sz <= args.max_backtracking_n:
                    bt = tsp_backtracking(sub_dist, start=0, time_limit=args.time_limit, max_nodes_explore=args.bt_node_limit)
                    bt_route_idx = bt["best_route"]
                    bt_route_names = [sub_names[i] for i in bt_route_idx] if bt_route_idx else None

                    results.append({
                        "experiment_id": experiment_id,
                        "instance_id": instance_id,
                        "n": sz,
                        "repeat": r,
                        "seed": run_seed,
                        "algorithm": "backtracking",
                        "best_cost_m": float(bt["best_cost"]),
                        "runtime_s": float(bt["runtime_s"]),
                        "cutoff": bool(bt["cutoff"]),
                        "nodes_explored": int(bt["nodes_explored"]),
                        "pruned_count": int(bt["pruned_count"]),
                        "iteration_best_found": bt["iteration_best_found"],
                        "peak_memory_mb": float(bt["peak_memory_mb"]),
                        "route_idx": bt_route_idx,
                        "route_names": bt_route_names,
                        "nodes_used": ";".join(sub_names),
                        "bt_node_limit": int(args.bt_node_limit),
                    })
                else:
                    if r == 0:
                        print(f"[INFO] Backtracking diskip untuk n={sz} (>{args.max_backtracking_n})")

    # simpan hasil
    df = pd.DataFrame(results)
    raw_path = os.path.join(args.out_dir, f"{experiment_id}_results_raw.csv")
    df.to_csv(raw_path, index=False)

    # summary sederhana: mean runtime per n-algo + cutoff rate
    summary = (
        df.groupby(["instance_id", "n", "algorithm"], as_index=False)
          .agg(runtime_mean_s=("runtime_s", "mean"),
               runtime_std_s=("runtime_s", "std"),
               best_cost_min_m=("best_cost_m", "min"),
               cutoff_rate=("cutoff", "mean"))
    )
    sum_path = os.path.join(args.out_dir, f"{experiment_id}_summary.csv")
    summary.to_csv(sum_path, index=False)

    print("\n================ DONE ================")
    print(f"Raw results   : {raw_path}")
    print(f"Summary       : {sum_path}")
    print("======================================")


if __name__ == "__main__":
    main()
