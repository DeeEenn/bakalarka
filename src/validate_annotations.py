import os
import numpy as np

def validate_annotations():
    """
    Ověří, že počet labelů odpovídá počtu snímků v NPY souborech
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    features_dir = os.path.join(project_root, "data", "features")
    labels_dir = os.path.join(project_root, "data", "labels")
    
    print("\n" + "="*80)
    print("VALIDACE SYNCHRONIZACE: Features vs Labels")
    print("="*80)
    
    errors = []
    validated = 0
    
    for root, dirs, files in os.walk(features_dir):
        for file in files:
            if file.endswith(".npy"):
                npy_path = os.path.join(root, file)
                
                # Najdi odpovídající label file
                rel_path = os.path.relpath(root, features_dir)
                label_dir = os.path.join(labels_dir, rel_path)
                label_path = os.path.join(label_dir, os.path.splitext(file)[0] + ".txt")
                
                if not os.path.exists(label_path):
                    errors.append(f"❌ {file}: Chybí anotace ({label_path})")
                    continue
                
                # Načti počet snímků
                features = np.load(npy_path)
                num_features = features.shape[0]
                
                with open(label_path, 'r') as f:
                    labels = [int(line.strip()) for line in f if line.strip()]
                num_labels = len(labels)
                
                # Validace
                if num_features == num_labels:
                    validated += 1
                    print(f"✓ {file}: {num_features} snímků = {num_labels} labelů")
                else:
                    errors.append(f"❌ {file}: {num_features} snímků ≠ {num_labels} labelů (rozdíl: {abs(num_features - num_labels)})")
    
    print("\n" + "="*80)
    print("VÝSLEDKY VALIDACE")
    print("="*80)
    print(f"✓ Validováno: {validated} souborů")
    print(f"❌ Chyby: {len(errors)} souborů")
    
    if errors:
        print("\nCHYBNÉ SOUBORY:")
        for err in errors:
            print(f"  {err}")
    else:
        print("\n🎉 VŠECHNY ANOTACE SEDÍ S FEATURES!")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    validate_annotations()
