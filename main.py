import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Functie pentru incarcare date dintr-un fisier .pkl
def load_data_from_pickle(file_path):
    try:
        # Incarca datele folosind pickle
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return np.array(data)
    except:
        print("Eroare la incarcarea fisierului:", file_path)
        return None

def monte_carlo_simulation_batches(data, total_iterations=3_825_385_140, batch_size=1_000_000, hist_bins=100):
    if data is None or len(data) == 0:
        return None

    n_batches = total_iterations // batch_size
    gmin = float("inf")  # Valoarea minima globala
    gmax = float("-inf")  # Valoarea maxima globala
    dmin, dmax = data.min(), data.max()  # Valoarea minima si maxima din date
    hrange = (dmin - 1, dmax + 1)  # Interval pentru histograma
    hist_counts = np.zeros(hist_bins, dtype=np.float64)

    for i in range(n_batches):
        idx = np.random.randint(0, len(data), size=batch_size)
        vals = data[idx]
        mn, mx = vals.min(), vals.max()
        if mn < gmin: gmin = mn
        if mx > gmax: gmax = mx
        c, _ = np.histogram(vals, bins=hist_bins, range=hrange)
        hist_counts += c

    return {"hist_counts": hist_counts, "hist_range": hrange, "hist_bins": hist_bins, "min": gmin, "max": gmax}

def plot_histogram_only(hist_counts, hist_range, hist_bins, output_path, title):
    plt.figure(figsize=(10, 6))
    hmin, hmax = hist_range
    edges = np.linspace(hmin, hmax, hist_bins + 1)
    plt.bar(edges[:-1], hist_counts, width=np.diff(edges), alpha=0.7, color="blue", edgecolor="black")  # Bara histogramelor

    centers = 0.5 * (edges[:-1] + edges[1:])
    total = hist_counts.sum()

    # Calculeaza media si deviatia standard daca histograma nu este goala
    if total > 0:
        mean = (centers * hist_counts).sum() / total
        var = ((centers - mean) ** 2 * hist_counts).sum() / total
        std = np.sqrt(var)
        left = mean - 2 * std  # Limita stanga
        right = mean + 2 * std  # Limita dreapta
        plt.axvspan(left, right, color="green", alpha=0.2, label="Interval 95% (±2σ)")

    plt.title(title, fontsize=15)
    plt.xlabel("Valoare simulata", fontsize=12)
    plt.ylabel("Frecventa", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    results_dir = "results"
    output_dir = "results_png"
    os.makedirs(output_dir, exist_ok=True)

    total_iterations = 191_269_257
    batch_size = 1_000_000
    hist_bins = 25

    if not os.path.isdir(results_dir):
        print(f"Directorul '{results_dir}' nu exista.")
        exit(1)

    pkl_files = [f for f in os.listdir(results_dir) if f.endswith(".pkl")]
    if not pkl_files:
        print(f"Niciun fisier .pkl gasit in '{results_dir}'")
        exit(0)

    for pkl_file in pkl_files:
        path = os.path.join(results_dir, pkl_file)
        print(f"\nProcesare {path}")
        data = load_data_from_pickle(path)
        if data is None:
            print("Incarcare esuata")
            continue
        print(f"Ruleaza Monte Carlo ({total_iterations:,} iteratii, batch={batch_size:,})")
        res = monte_carlo_simulation_batches(data, total_iterations, batch_size, hist_bins)
        if not res:
            print("Niciun rezultat")
            continue
        out_name = os.path.splitext(pkl_file)[0] + "_hist.png"
        out_path = os.path.join(output_dir, out_name)
        print(f"Salvat in{out_path}")
        plot_histogram_only(res["hist_counts"], res["hist_range"], res["hist_bins"], out_path, pkl_file)
