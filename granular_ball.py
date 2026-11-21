# -*- coding: utf-8 -*-
from collections import Counter
import warnings
import numpy as np
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")


class GranularBall:
    def __init__(self, data):
        self.data = data
        self.data_no_label = data[:, :-2]
        self.num, self.dim = self.data_no_label.shape
        self.center = self.data_no_label.mean(axis=0)
        self.label, self.purity, self.r = self._compute_attributes()

    def _compute_attributes(self):
        counts = Counter(self.data[:, -2])
        label = max(counts, key=counts.get)
        purity = counts[label] / self.num
        r = np.mean(np.linalg.norm(self.data_no_label - self.center, axis=1))
        return label, purity, r

    def split_2balls(self):
        clu = KMeans(n_clusters=2, n_init=10, random_state=0).fit(self.data_no_label)
        labels = clu.labels_
        if np.any(labels == 0) and np.any(labels == 1):
            return GranularBall(self.data[labels == 0]), GranularBall(self.data[labels == 1])
        return GranularBall(self.data[0:1]), GranularBall(self.data[1:])


class GBList:
    def __init__(self, data):
        self.data = data
        self.granular_balls = [GranularBall(self.data)]

    def init_granular_balls(self, purity=1.0, min_sample=1):
        ll = len(self.granular_balls)
        i = 0
        while i < ll:
            if self.granular_balls[i].purity < purity and self.granular_balls[i].num > min_sample:
                b1, b2 = self.granular_balls[i].split_2balls()
                self.granular_balls[i] = b1
                self.granular_balls.append(b2)
                ll += 1
            else:
                i += 1
        self.data = np.vstack([ball.data for ball in self.granular_balls])

    def del_ball(self, purity=0.0, num_data=0):
        filtered = [
            ball for ball in self.granular_balls
            if ball.purity >= purity and ball.num >= num_data
        ]
        if any(b.label == 1 for b in filtered) and any(b.label != 1 for b in filtered):
            self.granular_balls = filtered
            self.data = np.vstack([ball.data for ball in self.granular_balls])


def get_gb_details(data, gamma=0.85, min_samples=2):
    num = data.shape[0]
    index = np.arange(num).reshape(num, 1)
    data_indexed = np.hstack((data, index))

    gb_list = GBList(data_indexed)
    gb_list.init_granular_balls(purity=gamma)
    gb_list.del_ball(num_data=min_samples)

    # Order positive balls first to match S_hat = blkdiag(S_hat_+, S_hat_-)
    pos_balls = [b for b in gb_list.granular_balls if b.label == 1]
    neg_balls = [b for b in gb_list.granular_balls if b.label != 1]

    return [
        {
            'center': ball.center,
            'r': ball.r,
            'label': 1 if ball.label == 1 else -1,
            'num': ball.num,
            'purity': ball.purity
        }
        for ball in (pos_balls + neg_balls)
    ]