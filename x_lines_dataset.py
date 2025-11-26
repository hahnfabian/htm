from torch.utils.data import Dataset, DataLoader
import numpy as np 
import torch


img_size = (128, 128)
angular_sep = (20, 160)
line_len = 10
batch_size = 64


class LinePairsDataset(Dataset):
    def __init__(self, dataset_size=1000, num_pairs=8):
        self.dataset_size = dataset_size
        self.num_pairs = num_pairs

        self.data = [self._create_sample() for _ in range(dataset_size)]

    def _create_line_pair(self):
        img_w, img_h = img_size  # assuming img_size = (width, height)
        cx = np.random.rand() * img_w
        cy = np.random.rand() * img_h
        theta1 = np.deg2rad(np.random.randint(0, 360))
        theta2 = theta1 + np.deg2rad(np.random.randint(*angular_sep))
        half_len = line_len / 2

        dx1, dy1 = half_len * np.cos(theta1), half_len * np.sin(theta1)
        dx2, dy2 = half_len * np.cos(theta2), half_len * np.sin(theta2)

        line1 = [cx + dx1, cy + dy1, cx - dx1, cy - dy1]
        line2 = [cx + dx2, cy + dy2, cx - dx2, cy - dy2]

        # Clamp coordinates to image boundaries
        line1 = np.clip(line1, [0, 0, 0, 0], [img_w, img_h, img_w, img_h])
        line2 = np.clip(line2, [0, 0, 0, 0], [img_w, img_h, img_w, img_h])

        lines = np.round([line1, line2], 4)

        return np.array(lines, dtype=np.float32)

    def _create_sample(self):
        pairs = []
        for _ in range(self.num_pairs):
            pair = self._create_line_pair()
            pairs.append(pair)
        
        return np.concatenate(pairs, axis=0)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        tokens = torch.tensor(self.data[idx], dtype=torch.float32)
        tokens = (tokens / 64.0) - 1.0
        return tokens



def get_x_lines_dataset(dataset_size=1000, num_pairs=8):
    dataset = LinePairsDataset(dataset_size, num_pairs)

    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
