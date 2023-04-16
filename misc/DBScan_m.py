import torch
from DTM import euclidean_dist


class DBScan:
    def __init__(self):
        self.outliers = []
        self.radius = []
        self.unvisited = torch.arange(0, 1)
        self.N = 0

    def scan(self, centroids, features):
        self.unvisited = torch.arange(0, features.size(0))
        for i, centroid in centroids:
            self.extend_cls(cls=i, centroid=centroid.reshape(1, centroids.size(1)), features=features)

        features_u = features[self.unvisited]
        features_o = torch.cat(self.outliers, 0)
        features_o = features_o.view(-1, features.size(1))
        features_u = torch.cat([features_u, features_o], dim=0)
        radius_avg = sum(self.radius) / len(self.radius)
        for feature in features_u:
            dist = euclidean_dist(feature, features)
            visited = features[dist <= radius_avg]
            if visited.size(0) >= self.N:
                self.extend_outliers(radius_avg, feature, features_u)
        # TODO
        self.outliers.append([_ for _ in features[self.unvisited]])
        return self.unvisited

    def extend_cls(self, cls, centroid, features):
        features_u = features[self.unvisited]
        dist = euclidean_dist(centroid, features_u)
        self.unvisited = self.unvisited[dist > self.radius[cls]]
        visited = features_u[dist <= self.radius[cls]]
        for centroid_n in visited:
            self.extend_cls(cls=cls, centroid=centroid_n.reshape(1, visited.size(1)), features=features)

    def extend_outliers(self, radius, centroid, features):
        dist = euclidean_dist(centroid, features)
        self.unvisited = self.unvisited[dist[:self.unvisited.size(0)] > radius]
        visited = features[dist <= radius]
        features = features[dist > radius]
        for centroid_n in visited:
            self.extend_outliers(radius=radius, centroid=centroid_n.reshape(1, visited.size(1)), features=features)

    def set_r_n(self, radius, N):
        self.radius = radius
        self.N = N


if __name__ == "__main__":
    a = torch.ones(5)
