import torch 
from torch.utils.data import Dataset
import numpy as np
import os

class InhalerDataset(Dataset):
    def __init__(self, features_dir, labels_dir, max_len=2000):
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.max_len = max_len
        self.data_list = self._get_data_list()

    def _get_data_list(self):
        samples = []
        for root, _, files in os.walk(self.features_dir):
            for file in files:
                if file.endswith(".npy"):
                    # Zachování relativní cesty pro nalezení labelu
                    rel_path = os.path.relpath(os.path.join(root, file), self.features_dir)
                    label_file = os.path.join(self.labels_dir, rel_path.replace(".npy", ".txt"))
                    
                    if os.path.exists(label_file):
                        samples.append((os.path.join(root, file), label_file))
        return samples
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        feature_path, label_path = self.data_list[idx]

    # Načtení dat (243, T)
        features = np.load(feature_path).T 
        labels = np.loadtxt(label_path, dtype=np.int64)

        T = features.shape[1]
        
        # Ošetření délky (Padding)
        if T < self.max_len:
            pad_width = self.max_len - T
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
            # Label padding: 0 (Klid) nebo -100 pro ignorování v Loss
            labels = np.pad(labels, (0, pad_width), mode='constant', constant_values=-100)
        else:
            features = features[:, :self.max_len]
            labels = labels[:self.max_len]

        T_eff = min(T, self.max_len)

        return torch.from_numpy(features).float(), torch.from_numpy(labels).long(), T_eff