import copy
import torch


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


class DBScan:
    def __init__(self):
        self.outliers = []
        self.radius = []
        self.unvisited = torch.arange(0, 1)
        self.N = 0

    def scan(self, centroids, features):
        self.unvisited = torch.arange(0, features.size(0))
        for i, centroid in enumerate(centroids):
            self.extend_cls(cls=i, centroid=centroid.reshape(1, centroids.size(1)), features=features)

        features_u = features[self.unvisited]
        if len(self.outliers):
            features_o = torch.cat(self.outliers, 0)
            features_o = features_o.view(-1, features.size(1))
            features_u = torch.cat([features_u, features_o], dim=0)
        radius_avg = sum(self.radius) / len(self.radius)
        feature_i = copy.deepcopy(features_u)
        for feature in feature_i:
            feature = feature.reshape(1, features_u.size(1))
            dist = euclidean_dist(feature, features_u)
            visited = features_u[dist.reshape(dist.size(1)) <= radius_avg]
            if visited.size(0) >= self.N:
                features_u = self.extend_outliers(radius_avg, feature, features_u)

        self.outliers = [_ for _ in features_u]
        # print(len(self.outliers))
        return self.unvisited

    def extend_cls(self, cls, centroid, features):
        features_u = features[self.unvisited]
        dist = euclidean_dist(centroid, features_u)
        self.unvisited = self.unvisited[dist.reshape(dist.size(1)) > self.radius[cls]]
        visited = features_u[dist.reshape(dist.size(1)) <= self.radius[cls]]
        for centroid_n in visited:
            self.extend_cls(cls=cls, centroid=centroid_n.reshape(1, visited.size(1)), features=features)

    def extend_outliers(self, radius, centroid, features):
        dist = euclidean_dist(centroid, features)
        dist = dist.reshape(dist.size(1))
        self.unvisited = self.unvisited[dist[:self.unvisited.size(0)] > radius]
        visited = features[dist <= radius]
        features = features[dist > radius]
        for centroid_n in visited:
            features = self.extend_outliers(radius=radius, centroid=centroid_n.reshape(1, visited.size(1)), features=features)
        return features

    def set_r_n(self, radius, N):
        self.radius = radius * 2
        self.N = N / 10

    def get_outliers(self):
        self.outliers = []
        return len(self.outliers)


if __name__ == "__main__":
    a = torch.ones(5)
