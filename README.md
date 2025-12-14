# Traveling Salesman Problem (TSP) Solver

## Deskripsi
Proyek ini menyediakan implementasi algoritma **Backtracking** dan **Branch & Bound** untuk menyelesaikan **Travelling Salesman Problem (TSP)**.  
Kode ini mendukung analisis performa runtime, jumlah node yang dieksplorasi/diekspansi, penggunaan memori, dan ringkasan hasil eksperimen. Data input berupa file CSV dengan koordinat lokasi.

---

## Fitur dan Cara Penggunaan

1. **Import dataset CSV**  
   - Harus memiliki kolom: `id`, `nama_tempat`, `latitude`, `longitude`.  
   - Validasi data otomatis: menghapus nilai lat/lon yang tidak valid dan duplikat.  

2. **Algoritma TSP**  
   - **Backtracking**: optimal untuk jumlah node kecil (`n <= max_backtracking_n`).  
   - **Branch & Bound**: cocok untuk jumlah node lebih besar, menggunakan reduksi matriks dan bound untuk memangkas pohon pencarian.  

3. **Analisis hasil eksperimen**  
   - Ringkasan runtime rata-rata ± std dan CI95.  
   - Node yang dieksplorasi/diekspansi.  
   - Peak memory usage.  
   - Cutoff rate (jika melebihi batas waktu atau node).  
   - Visualisasi: runtime vs jumlah node, boxplot runtime per algoritma, rata-rata node dieksplorasi/diekspansi ± std.

4. **Utility**  
   - Fungsi `haversine()` untuk menghitung jarak antar koordinat.  
   - Fungsi `summarize()` untuk menghitung mean, std, dan CI95.  
   - Informasi hardware (CPU, RAM, GPU).
