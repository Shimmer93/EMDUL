import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from copy import deepcopy
from itertools import chain

def select_split_subset(split_indices, ratio, subset_seed=0):
    split_indices = list(split_indices)
    if ratio >= 1:
        return split_indices, []

    subset_len = int(len(split_indices) * ratio)
    if subset_len <= 0:
        return [], split_indices

    # The development runs used the first fraction of the stored sequence IDs.
    if subset_seed is None:
        return split_indices[:subset_len], split_indices[subset_len:]

    rng = np.random.default_rng(subset_seed)
    subset_positions = set(rng.choice(len(split_indices), size=subset_len, replace=False).tolist())
    subset = [idx for pos, idx in enumerate(split_indices) if pos in subset_positions]
    remainder = [idx for pos, idx in enumerate(split_indices) if pos not in subset_positions]
    return subset, remainder

class BaseDataset(Dataset):
    def __init__(self, data_path, transform=None, split='train', ratio=1, subset_seed=0,
                 trim_tail_frames=0):
        super().__init__()
        
        self.data_path = data_path
        self.transform = transform

        with open(data_path, 'rb') as f:
            self.all_data = pickle.load(f)

        if isinstance(split, str):
            self.split_ = self.all_data['splits'][split]
        elif isinstance(split, list):
            self.split_ = [self.all_data['splits'][s] for s in split]
            self.split_ = list(chain(*self.split_))

        self.split, self.split_rest = select_split_subset(self.split_, ratio, subset_seed)
        self.data = [self.all_data['sequences'][i] for i in self.split]
        if trim_tail_frames < 0:
            raise ValueError('trim_tail_frames must be nonnegative')
        self.seq_lens = [max(0, len(seq['keypoints']) - trim_tail_frames) for seq in self.data]

    def __len__(self):
        return int(np.sum(self.seq_lens))

    def _build_sample(self, sequence, data_path, seq_idx, frame_idx, global_idx):
        sample = deepcopy(sequence)
        sample['sequence_index'] = seq_idx
        sample['global_index'] = global_idx
        sample['index'] = frame_idx
        sample['centroid'] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        sample['radius'] = 1.0
        sample['scale'] = 1.0
        sample['translate'] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        sample['rotation_matrix'] = np.eye(3, dtype=np.float32)
        sample['dataset_name'] = data_path.split('/')[-1].split('.')[0]
        return sample
    
    def __getitem__(self, idx):
        seq_idx = 0
        global_idx = idx
        while idx >= self.seq_lens[seq_idx]:
            idx -= self.seq_lens[seq_idx]
            seq_idx += 1
        sample = self._build_sample(self.data[seq_idx], self.data_path, seq_idx, idx, global_idx)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
    
    @staticmethod
    def collate_fn(batch):
        batch_data = {}
        keys = ['point_clouds', 'keypoints', 'centroid', 'radius', 'sequence_index', 'index', 'global_index']
        for key in keys:
            batch_data[key] = torch.stack([sample[key] for sample in batch], dim=0)

        return batch_data
