import argparse
import json
import os
import random
from datetime import datetime

import pandas as pd


def load_and_clean_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    required = {"id", "nama_tempat", "latitude", "longitude"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV harus punya kolom: {required}. Ditemukan: {df.columns.tolist()}")

    df = df[["id", "nama_tempat", "latitude", "longitude"]].copy()
    df = df.dropna(subset=["latitude", "longitude"])

    # valid range
    df = df[df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)].copy()

    # buang duplikat koordinat
    df = df.drop_duplicates(subset=["latitude", "longitude"]).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("Hasil cleaning kosong. Cek isi CSV (lat/lon).")

    return df


def write_instance_json(out_path: str, instance_id: str, nodes: list, start_index: int = 0):
    payload = {
        "problem": "tsp_geo",
        "instance_id": instance_id,
        "description": "TSP geo instance generated from CSV (lat/lon).",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "start_index": int(start_index),
        "n": int(len(nodes)),
        "nodes": nodes,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path CSV (kolom: id,nama_tempat,latitude,longitude)")
    ap.add_argument("--out_dir", default="data", help="Folder output JSON")
    ap.add_argument("--count", type=int, default=5, help="Jumlah instance (G01..)")
    ap.add_argument("--n", type=int, default=12, help="Jumlah node per instance (akan di-auto adjust jika > dataset)")
    ap.add_argument("--seed", type=int, default=12345, help="Base seed")
    ap.add_argument("--prefix", default="tsp_G", help="Prefix filename (tsp_G01.json)")
    ap.add_argument("--start_index", type=int, default=0, help="Start node index dalam instance (default 0)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_and_clean_csv(args.csv)

    available = len(df)
    if args.n > available:
        print(f"[INFO] Dataset hanya punya {available} node, jadi --n diubah otomatis dari {args.n} -> {available}")
    n_use = min(args.n, available)

    # start_index harus valid
    if args.start_index < 0 or args.start_index >= n_use:
        print(f"[INFO] start_index={args.start_index} di luar range [0..{n_use-1}], diset jadi 0")
        args.start_index = 0

    for k in range(1, args.count + 1):
        rng = random.Random(args.seed + k)
        picked = rng.sample(range(available), n_use)
        sub = df.iloc[picked].reset_index(drop=True)

        # reindex id 0..n-1
        nodes = []
        for i in range(len(sub)):
            nodes.append(
                {
                    "id": int(i),
                    "name": str(sub.loc[i, "nama_tempat"]),
                    "latitude": float(sub.loc[i, "latitude"]),
                    "longitude": float(sub.loc[i, "longitude"]),
                }
            )

        instance_id = f"{args.prefix}{k:02d}"
        out_path = os.path.join(args.out_dir, f"{instance_id}.json")
        write_instance_json(out_path, instance_id, nodes, start_index=args.start_index)
        print(f"[OK] wrote {out_path} (n={n_use})")

    # dummy kecil buat test cepat
    dummy_n = min(8, available)
    rng = random.Random(args.seed)
    picked = rng.sample(range(available), dummy_n)
    sub = df.iloc[picked].reset_index(drop=True)
    nodes = [
        {
            "id": int(i),
            "name": str(sub.loc[i, "nama_tempat"]),
            "latitude": float(sub.loc[i, "latitude"]),
            "longitude": float(sub.loc[i, "longitude"]),
        }
        for i in range(len(sub))
    ]
    write_instance_json(os.path.join(args.out_dir, "tsp_dummy.json"), "tsp_dummy", nodes, start_index=0)
    print("[OK] wrote data/tsp_dummy.json")


if __name__ == "__main__":
    main()
