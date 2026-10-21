
from scipy.sparse import csr_matrix
import numpy as np
from tqdm import tqdm
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os, shutil, json
import sklearn
import itertools

class Estimator():
    def __init__(self, train_df):
        self.train_df = train_df
        self.users = np.sort(train_df["user_id"].unique())
        self.apps  = np.sort(train_df["app_id"].unique())
        self.u_idx = {u: i for i, u in enumerate(self.users)}
        self.i_idx = {a: i for i, a in enumerate(self.apps)}
    def get_csr_matrix(self, df):
        pos = df[df["is_recommended"] > 0]
        rows = pos["user_id"].map(self.u_idx).values
        cols = pos["app_id"].map(self.i_idx).values
        data = np.ones(len(pos), dtype=np.float32)
        R = csr_matrix((data, (rows, cols)), shape=(len(self.users), len(self.apps)))
        return R
    def precision(self, preds, true, k=10):
        intersection = set(preds) & set(true)
        return len(intersection) / k
    def ndcg_at_k(self, top_k_items, relevant_items, k=10):
        dcg = 0.0
        for rank, item in enumerate(top_k_items[:k], start=1):
            if item in relevant_items:
                dcg += 1.0 / np.log2(rank + 1)
        idcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(relevant_items), k) + 1))
        return dcg / idcg if idcg > 0 else 0.0
