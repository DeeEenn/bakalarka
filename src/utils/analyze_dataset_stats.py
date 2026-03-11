import os
import numpy as np

def analyze_stats(features_dir, labels_dir):
    stats = []
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    class_names = {0: "KLID", 1: "PRIPRAVA", 2: "ROZDEJCHANI", 3: "INHALACE", 4: "ZADRZENI", 5: "VYDECH"}
    
    abs_labels = os.path.abspath(labels_dir)
    print(f"Hledám popisky v: {abs_labels}")

    if not os.path.exists(abs_labels):
        print(f"CHYBA: Složka {abs_labels} neexistuje!")
        return

    for root, _, files in os.walk(abs_labels):
        for file in files:
            if file.endswith(".txt"):
                label_path = os.path.join(root, file)
                try:
                    labels = np.loadtxt(label_path, dtype=int)
                    if labels.size == 0: continue
                    
                    duration_frames = labels.size
                    stats.append(duration_frames)
                    
                    for label in labels:
                        if label in class_counts:
                            class_counts[label] += 1
                except Exception as e:
                    print(f"Chyba při čtení {file}: {e}")

    if not stats:
        print("Nebyla nalezena žádná data. Zkontroluj cesty k souborům .txt.")
        return

    total_frames = sum(stats)
    avg_len = np.mean(stats) / 30 
    
    print(f"\n--- STATISTIKA DATASETU ---")
    print(f"Celkový počet videí: {len(stats)}")
    print(f"Průměrná délka: {avg_len:.2f} s")
    print(f"Celkem snímků: {total_frames}")
    print("\n--- DISTRIBUCE TŘÍD ---")
    for k, v in class_counts.items():
        percentage = (v / total_frames) * 100
        print(f"{class_names[k]:<12}: {v/30:>7.2f} s ({percentage:>5.1f} %)")

if __name__ == "__main__":
    analyze_stats("../../data/features_enhanced", "../../data/labels")