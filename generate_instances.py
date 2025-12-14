# generate_instances.py
import pandas as pd
import numpy as np
from pathlib import Path

# Folder untuk menyimpan instance
BASE = Path("data")
BASE.mkdir(exist_ok=True)

def save_instance(name, df):
    """Simpan DataFrame sebagai CSV di folder data/"""
    df.to_csv(BASE / name, index=False)
    print(f"Instance saved: {name}")

def generate_instances(df_original, n_instances=15, n_subset=10, seed_start=100):
    """
    Generate beberapa instance acak dari dataset utama.
    
    Args:
        df_original: DataFrame asli berisi kolom ['id','nama_tempat','latitude','longitude']
        n_instances: jumlah instance yang ingin dibuat
        n_subset: jumlah node per instance; jika None, pakai semua
        seed_start: seed awal untuk reproducibility
    """
    for i in range(n_instances):
        seed = seed_start + i
        np.random.seed(seed)
        if n_subset is None or n_subset > len(df_original):
            n_subset_i = len(df_original)
        else:
            n_subset_i = n_subset
        
        indices = np.random.choice(df_original.index, size=n_subset_i, replace=False)
        df_instance = df_original.loc[indices].reset_index(drop=True)
        filename = f"tsp_instance_{i+1:02d}.csv"
        save_instance(filename, df_instance)

if __name__ == "__main__":
    # contoh: generate instance dari CSV master
    master_csv = "tsp_cluster_barat_uns.csv"
    df_master = pd.read_csv(master_csv)
    
    # Normalisasi kolom
    df_master = df_master.rename(columns={c: c.strip() for c in df_master.columns})
    required_cols = ["id","nama_tempat","latitude","longitude"]
    if not all(col in df_master.columns for col in required_cols):
        raise ValueError(f"CSV harus memiliki kolom: {required_cols}")
    
    # Buat 10 instance acak, masing-masing 10 node (ubah sesuai kebutuhan)
    generate_instances(df_master, n_instances=10, n_subset=10)
