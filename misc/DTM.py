import numpy as np

from misc.feature_pool import FeaturePool
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import Counter
from misc.DBScan_m import DBScan
from scipy.stats import norm


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


class Cluster(nn.Module):

    def __init__(self, queue_len, num_class, feature_len, dist_type='cos', knn_num=1):
        super(Cluster, self).__init__()
        self.scanner = DBScan()
        self.class_pool = FeaturePool(queue_len, num_class, feature_len)
        self.num_class = num_class
        self.centroids = torch.zeros((num_class, feature_len)).cuda()
        self.count = torch.zeros((num_class, 1))
        self.dist_type = dist_type
        self.num_class = num_class
        self.pred_top_N = torch.zeros((self.num_class, knn_num)).fill_(-1).cuda()
        self.fake_index = [[-1 for i in range(knn_num)] for i in range(self.num_class)]
        self.fake_labels = [i for i in range(self.num_class) for j in range(knn_num)]

        self.ulb_dest_len = 50000
        # self.thresh_warmup = True
        self.selected_label = torch.ones((self.ulb_dest_len,), dtype=torch.long, ) * -1
        self.classwise_acc = torch.zeros((self.num_class,))

    def init_(self, sample_len=None, pool_size=None):
        self.pred_top_N = torch.zeros((self.num_class, sample_len)).fill_(-1).cuda()
        self.fake_index = [[-1 for i in range(sample_len)] for i in range(self.num_class)]
        self.fake_labels = [i for i in range(self.num_class) for j in range(sample_len)]
        for i in range(self.num_class):
            self.class_pool.pool_size[i] = pool_size * 10

    def add_sample(self, feature, target):
        self.class_pool.add(feature, target)

    def get_centroid(self):
        for i in range(self.num_class):
            self.centroids[i] = self.class_pool.query(i)
        return self.centroids

    # def update_centroid(self):
    #     self.centroid_dist = euclidean_dist(self.centroids, self.centroids) + torch.eye(self.num_class,
    #                                                                                     self.num_class).cuda() * 1e10
    #     return bool

    def cosine_similarity(self, x, y):
        dist = float("inf")
        dist = torch.cosine_similarity(x, y, dim=1)
        return dist

    def new_data(self):
        return self.fake_index, self.fake_labels

    def forward(self, feature, pred, unlabeled_index):
        # self.update_centroid()
        self.get_centroid()
        # radius, N = self.class_pool.get_r_n(self.centroids)
        # self.scanner.set_r_n(radius, N)
        # outliers = self.scanner.scan(self.centroids, feature)
        if self.dist_type == 'cos':
            label = torch.zeros(feature.size(0)).fill_(-1)
            weight = torch.zeros(feature.size(0)).fill_(1)
            cos_f = torch.zeros(feature.size(0)).fill_(0)
            # mask = torch.zeros(feature.size(0)).fill_(0)
            select = torch.zeros(feature.size(0)).fill_(0)
            # euc_dist = euclidean_dist(feature, self.centroids)
            # euc_dist = euc_dist/euc_dist.sum(1,True)  #F.softmax(euc_dist,1)
            # min_euc, euc_id = euc_dist.min(1)  # euc_dist.topk(1,1,False,True)
            # assert len(min_euc) == len(feature)

            pred = F.softmax(pred, 1)
            scores, lbs = torch.max(pred, dim=1)
            pdf = []
            mean_cls = []
            cos_cls = [[] for _ in range(self.num_class)]

            for i, f in enumerate(feature, 0):
                f = f.unsqueeze(0)
                cos_dist = self.cosine_similarity(f, self.centroids)
                cos_max, label_idx = cos_dist.max(0)  # topk(1,0,True,True)  #top 2 max

                # if label_idx == euc_id[i] and cos_max > (0.85 * (self.classwise_acc[label_idx] / (2. - self.classwise_acc[label_idx]))):
                #     label[i] = label_idx
                #     # mask[i] = 1
                #
                select[i] = 1
                label[i] = label_idx
                cos_cls[label_idx].append(cos_max)
                cos_f[i] = cos_max

                min_tmp, min_idx = self.pred_top_N[label_idx].min(0)
                if scores[i] > min_tmp:
                    self.pred_top_N[label_idx][min_idx] = scores[i]
                    self.fake_index[label_idx][min_idx] = unlabeled_index[i]

            for cos in cos_cls:
                cos = torch.stack(cos)
                mean = torch.mean(cos).cpu()
                var = torch.var(cos).cpu()
                pdf_tmp = norm(mean, torch.sqrt(var)).pdf
                pdf.append(pdf_tmp)
                mean_cls.append(mean)

            for i in range(10):
                print(pdf[i](mean_cls[i].cpu()))

            for i in range(feature.size(0)):
                idx = int(label[i].item())
                if cos_f[i] < mean_cls[idx]:
                    weight[i] = pdf[idx](cos_f[i].cpu())


            # label[idx] = lbs[idx].cpu().float()
            # label[outliers] = -1
            # update
            if unlabeled_index[select == 1].nelement() != 0:
                self.selected_label[unlabeled_index[select == 1]] = label[select == 1].to(self.selected_label.dtype)
            self.update()
            return label, weight

    @torch.no_grad()
    def update(self, *args, **kwargs):
        pseudo_counter = Counter(self.selected_label.tolist())
        if max(pseudo_counter.values()) < self.ulb_dest_len:  # not all(5w) -1
            for i in range(self.num_class):
                self.classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())


