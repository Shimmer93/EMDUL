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
    def __init__(self, max_dist_threshold=0.1, min_dist_threshold=0.05, add_std=0.1, default_num_points=32, num_noisy_points=32):
        self.max_dist_threshold = max_dist_threshold
        self.min_dist_threshold = min_dist_threshold
        self.add_std = add_std
        self.default_num_points = default_num_points
        self.num_noisy_points = num_noisy_points

    def __call__(self, sample):
        if isinstance(sample['keypoints'], list):
            sample['keypoints'] = np.stack(sample['keypoints'])

        keypoints = sample['keypoints']
        point_clouds = sample['point_clouds']

        # Match ConvertToRefinedMMWavePointCloud by adding 8 bbox corners as anchors.
        pc_cat = np.concatenate(point_clouds, axis=0)
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
        keypoint_flow = keypoints_with_anchors[1:] - keypoints_with_anchors[:-1]

        new_pcs = []
        for t in range(T - 1):
            pc = point_clouds[t][:, :3]
            kp = keypoints_with_anchors[t][:, :3]
            kpf = keypoint_flow[t]
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

            mask = np.random.rand(pc.shape[0]) < prob
            if np.any(mask):
                new_pc = pc[mask]
            else:
                random_idxs = np.random.choice(pc.shape[0], self.default_num_points, replace=pc.shape[0] < self.default_num_points)
                new_pc = pc[random_idxs]
            new_pcs.append(new_pc)
    
        sample['point_clouds'] = new_pcs
        # Keep the source-frame skeletons, matching ConvertToRefinedMMWavePointCloud.
        sample['keypoints'] = sample['keypoints'][:-1]
        return sample
