import random
import numpy as np
import torch
from torch.autograd import Variable
from collections import deque


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class FeaturePool:
    def __init__(self, pool_size, num_class, feature_len):
        self.feature_len = feature_len
        self.pool_size = {x: pool_size for x in range(num_class)}
        self.num_class = num_class
        self.num_imgs = {x: 0 for x in range(num_class)}
        self.features = {x: deque() for x in range(num_class)}

    def add(self, features, target):
        for idx, (feature, label) in enumerate(zip(features.data, target)):
            # print(feature.size(),label)
            cls = label.item()
            if cls == -1:
                continue
            if self.num_imgs[cls] < self.pool_size[cls]:
                self.num_imgs[cls] = self.num_imgs[cls] + 1
                self.features[cls].append(feature)
            else:
                self.features[cls].popleft()
                self.features[cls].append(feature)

    def single_add(self, feature, label):
        cls = label.item()
        if cls == -1:
            return 0
        if self.num_imgs[cls] < self.pool_size[cls]:
            self.num_imgs[cls] = self.num_imgs[cls] + 1
            self.features[cls].append(feature)
        else:
            self.features[cls].popleft()
            self.features[cls].append(feature)

    def return_feature(self, cls_group):
        return_features = []
        return_labels = []
        for cls in cls_group:
            cls = cls.item()
            return_features.extend(list(self.features[cls]))
            return_labels.extend([cls for i in range(len(self.features[cls]))])
        return_features = torch.cat(return_features, 0)
        return_features = return_features.view(-1, self.feature_len)
        return return_features, return_labels

    def query(self, cls):
        if len(self.features[cls]) > self.pool_size[cls]:
            return_features = list(random.sample(self.features[cls], self.pool_size[cls]))
        else:
            return_features = list(self.features[cls])

        return_features = torch.cat(return_features, 0)
        return_features = return_features.view(-1, self.feature_len)

        cls_center = torch.mean(return_features, dim=0)
        return cls_center

    def get_r_n(self, centroids):
        radius = []
        N = 0
        for cls in range(self.num_class):
            return_features = list(self.features[cls])
            return_features = torch.cat(return_features, 0)
            return_features = return_features.view(-1, self.feature_len)
            dist = euclidean_dist(centroids[cls].reshape(1, centroids.size(1)), return_features)
            radius.append(torch.mean(dist))
            N += dist[dist < radius].size(0)
        return radius, N / self.num_class


if __name__ == '__main__':
    import random

    index = np.random.randint(0, 3, size=30)
    # index = random.sample(range(0, 54), 54)
    feature = torch.rand(30, 3).cuda()
    target = torch.Tensor(index).cuda().long()
    pred = torch.randn(30, 3).cuda()

    pool = FeaturePool(50, 3, feature.size()[1])
    pool.add(feature, target, pred)
    #
    # print(pool.features)
    # print(pool.num_imgs)
    print(pool.query(0, 'weight'))
    print(pool.query(0, 'mean'))
