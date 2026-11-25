import numpy as np

class ConvertToMMWavePointCloud():
    def __init__(self, max_dist_threshold=0.1, add_std=0.1, default_num_points=32, num_noisy_points=32):
        self.max_dist_threshold = max_dist_threshold
        self.add_std = add_std
        self.default_num_points = default_num_points
        self.num_noisy_points = num_noisy_points

    def __call__(self, sample):
        if isinstance(sample['keypoints'], list):
            sample['keypoints'] = np.stack(sample['keypoints'])

        kps0 = sample['keypoints'][:-1]
        kps1 = sample['keypoints'][1:]
        kps_dist = np.linalg.norm(kps0 - kps1, axis=-1)
        dist_threshold = self.max_dist_threshold
        mask = kps_dist > dist_threshold

        new_pcs = []
        for i in range(len(sample['keypoints']) - 1):
            pc0 = sample['point_clouds'][i]
            
            num_points = 0
            new_pc = []
            for j in range(sample['keypoints'].shape[1]):
                pc0_j = pc0[sample['point_clouds'][i][..., -1] == j+1]
                if mask[i, j]:
                    new_pc.append(pc0_j)
                    num_points += len(pc0_j)

            if num_points == 0:
                if pc0.shape[0] == 0:
                    new_pc.append(np.zeros((self.default_num_points, pc0.shape[-1]), dtype=pc0.dtype))
                else:
                    random_idxs = np.random.choice(pc0.shape[0], self.default_num_points, replace=pc0.shape[0] < self.default_num_points)
                    new_pc.append(pc0[random_idxs])
            new_pc = np.concatenate(new_pc)
            new_pcs.append(new_pc)

        sample['point_clouds'] = new_pcs
        sample['keypoints'] = sample['keypoints'][:-1]
        return sample
    
class FlowBasedPointFiltering():
    def __init__(self, max_dist_threshold=0.1, min_dist_threshold=0.05, add_std=0.1, default_num_points=32, num_noisy_points=32,
                 source_frame=False, max_static_keep=0.0, include_keypoint_bounds=True,
                 fallback_replace=False):
        if not 0 <= max_static_keep <= 1:
            raise ValueError('max_static_keep must be a probability between 0 and 1')
        self.max_dist_threshold = max_dist_threshold
        self.min_dist_threshold = min_dist_threshold
        self.add_std = add_std
        self.default_num_points = default_num_points
        self.num_noisy_points = num_noisy_points
        self.source_frame = source_frame
        self.max_static_keep = max_static_keep
        self.include_keypoint_bounds = include_keypoint_bounds
        self.fallback_replace = fallback_replace

    def __call__(self, sample):
        if isinstance(sample['keypoints'], list):
            sample['keypoints'] = np.stack(sample['keypoints'])

        keypoints = sample['keypoints']
        point_clouds = sample['point_clouds']

        # Match ConvertToRefinedMMWavePointCloud by adding 8 bbox corners as anchors.
        bounds = [pc[:, :3] for pc in point_clouds if len(pc)]
        if self.include_keypoint_bounds or not bounds:
            bounds.append(keypoints.reshape(-1, 3))
        pc_cat = np.concatenate(bounds, axis=0)
        min_coords = np.min(pc_cat[:, :3], axis=0)
        max_coords = np.max(pc_cat[:, :3], axis=0)
        anchor_points = np.array(
            np.meshgrid(
                [min_coords[0], max_coords[0]],
                [min_coords[1], max_coords[1]],
                [min_coords[2], max_coords[2]],
            )
        ).T.reshape(-1, 3)
        keypoints_with_anchors = np.concatenate(
            [keypoints, anchor_points[np.newaxis, :, :].repeat(keypoints.shape[0], axis=0)],
            axis=1,
        )

        T = keypoints_with_anchors.shape[0]
        flow_thres = np.random.uniform(self.min_dist_threshold, self.max_dist_threshold)
        # Vary static-return sensitivity across synthetic radar clips.
        static_keep = np.random.uniform(0, self.max_static_keep) if self.max_static_keep else 0.0
        keypoint_flow = keypoints_with_anchors[1:] - keypoints_with_anchors[:-1]

        new_pcs = []
        for pair in range(T - 1):
            frame = pair if self.source_frame else pair + 1
            pc = point_clouds[frame][:, :3]
            kp = keypoints_with_anchors[frame][:, :3]
            kpf = keypoint_flow[pair]
            if pc.shape[0] == 0:
                new_pcs.append(np.zeros((self.default_num_points, 3), dtype=keypoints.dtype))
                continue
            pc_expanded = pc[:, np.newaxis, :]
            kp_expanded = kp[np.newaxis, :, :]
            kpf_expanded = kpf[np.newaxis, :, :]

            pairwise_distances = np.linalg.norm(pc_expanded - kp_expanded, axis=-1)

            eps = 1e-8
            weights = 1.0 / (pairwise_distances + eps)
            weights_normalized = weights / (weights.sum(axis=-1, keepdims=True) + eps)

            weights_expanded = weights_normalized[:, :, np.newaxis]
            pcf = (weights_expanded * kpf_expanded).sum(axis=1)
            pcf = np.linalg.norm(pcf, axis=-1)
            prob = np.clip(pcf / flow_thres, 0, 1)
            prob = static_keep + (1 - static_keep) * prob

            mask = np.random.rand(pc.shape[0]) < prob
            if np.any(mask):
                new_pc = pc[mask]
            else:
                random_idxs = np.random.choice(pc.shape[0], self.default_num_points,
                                               replace=self.fallback_replace or pc.shape[0] < self.default_num_points)
                new_pc = pc[random_idxs]
            new_pcs.append(new_pc)
    
        sample['point_clouds'] = new_pcs
        sample['keypoints'] = keypoints[:-1] if self.source_frame else keypoints[1:]
        return sample
