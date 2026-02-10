import numpy as np
import os

def normalize_skeleton(data):
    """
    Provede prostorovou normalizaci na cele video.
    data: tvar (pocet_snimku, 132)
    """
    # Prevedeni na (pocet_snimku, 33 bodu, 4 hodnoty) pro snazsi pocitani
    sequence = data.reshape(-1, 33, 4)
    normalized_sequence = np.zeros_like(sequence)

    for f in range(len(sequence)): # Opraveno: přidána dvojtečka
        frame = sequence[f]

        # bod 11 je leve rameno, 12 prave
        # vypocitame stred mezi rameny
        shoulder_mid_x = (frame[11, 0] + frame[12, 0]) / 2
        shoulder_mid_y = (frame[11, 1] + frame[12, 1]) / 2
        shoulder_mid_z = (frame[11, 2] + frame[12, 2]) / 2

        # vypocitani vzdalenosti ramen pro skalovani
        shoulder_dist = np.sqrt((frame[11, 0] - frame[12, 0])**2 + 
                               (frame[11, 1] - frame[12, 1])**2 + 
                               (frame[11, 2] - frame[12, 2])**2)

        # osetreni nulove vzdalenosti
        if shoulder_dist == 0:
            shoulder_dist = 1.0
        
        # provedeme posun a skalovani u vsech bodu
        for i in range(33):
            # centrovani
            normalized_sequence[f, i, 0] = (frame[i, 0] - shoulder_mid_x) / shoulder_dist
            normalized_sequence[f, i, 1] = (frame[i, 1] - shoulder_mid_y) / shoulder_dist
            normalized_sequence[f, i, 2] = (frame[i, 2] - shoulder_mid_z) / shoulder_dist
            # visibility bez zmeny
            normalized_sequence[f, i, 3] = frame[i, 3]

    # vratime zpet do puvodniho tvaru
    return normalized_sequence.reshape(-1, 132)

def proces_all_features(input_root, output_root):
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith(".npy"):
                input_path = os.path.join(root, file)

                # nacteme data
                raw_data = np.load(input_path)

                # spustime normalizaci
                norm_data = normalize_skeleton(raw_data)

                # ulozime do nove slozky
                rel_path = os.path.relpath(root, input_root)
                target_dir = os.path.join(output_root, rel_path)
                os.makedirs(target_dir, exist_ok=True)

                np.save(os.path.join(target_dir, file), norm_data)
                print(f"Normalizovano: {file}") # Opraveno: print

if __name__ == "__main__":
    # Získáme absolutní cestu ke skriptu a od ní odvozujeme cesty k datům
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_dir = os.path.join(project_root, "data", "features")
    output_dir = os.path.join(project_root, "data", "features_norm")
    
    proces_all_features(input_dir, output_dir)
    print("Vsechna data byla centrovana a normalizovana")