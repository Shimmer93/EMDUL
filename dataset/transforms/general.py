import numpy as np
from copy import deepcopy

class MultipleKeyAggregate():
    def __init__(self, transforms, ori_key, more_keys):
        self.transforms = transforms
        self.ori_key = ori_key
        self.more_keys = more_keys

    def __call__(self, sample):
        for t in self.transforms:
            random_state = np.random.get_state()
            for another_key in self.more_keys:
                another_sample = deepcopy(sample)
                another_sample[self.ori_key] = another_sample[another_key]
                np.random.set_state(random_state)
                another_sample = t(another_sample)
                sample[another_key] = another_sample[self.ori_key]
            np.random.set_state(random_state)
            sample = t(sample)
        return sample

class RandomApply():
    def __init__(self, transforms, prob):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, sample):
        if np.random.rand() < self.prob:
            for t in self.transforms:
                sample = t(sample)
        return sample

class ComposeTransform():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
