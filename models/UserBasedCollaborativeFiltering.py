
from Estimator import Estimator

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

class UserBasedCollaborativeFiltering(Estimator):
    def __init__(self, train_df):
        super().__init__(train_df)
    def train(self):
        self.R_train = self.get_csr_matrix(self.train_df)
        self.norms = np.sqrt(np.asarray(self.R_train.multiply(self.R_train).sum(axis=1))).ravel()
    def get_prediction(self, ind, k=10, regularization=0.0, k_neighbors=None):
        if self.norms[ind] == 0:
            return np.zeros(self.R_train.shape[1]), np.array([], dtype=int)

        r_u = self.R_train[ind]
        dot = np.asarray((self.R_train @ r_u.T).todense()).ravel()
        denom = self.norms * self.norms[ind] + regularization
        sim = np.where(denom > 0, dot / denom, 0.0)
        sim[ind] = 0.0

        if k_neighbors is not None and k_neighbors < len(sim):
            top = np.argpartition(-sim, k_neighbors)[:k_neighbors]
            weights = np.zeros_like(sim)
            weights[top] = sim[top]
        else:
            weights = sim

        nom = np.asarray(weights @ self.R_train).ravel()
        den = np.abs(weights).sum()
        scores = nom / den

        seen = self.R_train[ind].indices
        scores[seen] = -np.inf

        order = np.argsort(-scores)
        order = order[np.isfinite(scores[order])][:k]
        return scores, order
    def test(self, value_df, k=10, regularization=0.0, k_neighbors=None, debug=False, app_to_title=lambda x:x):
        R_val = self.get_csr_matrix(value_df)

        row_nnz = R_val.getnnz(axis=1)
        nonempty_rows = np.flatnonzero(row_nnz > 0)
        if debug:
            nonempty_rows = nonempty_rows[:5]
        total_precision = 0
        total_ndcg = 0
        total_els = 0
        for ind in nonempty_rows:
            true_items = [self.i_idx[x] for x in list(value_df[(value_df["user_id"] == self.users[ind]) & (value_df["is_recommended"] == True)]["app_id"])]
            if debug:
                train_items = [self.i_idx[x] for x in list(self.train_df[(self.train_df["user_id"] == self.users[ind]) & (self.train_df["is_recommended"] == True)]["app_id"])]
            #true_scores = R_val[ind].toarray().ravel()
            pred_scores, pred_items = self.get_prediction(ind, k, regularization, k_neighbors)
            if debug:
                print('Тренировка:', [app_to_title[a] for a in self.apps[train_items]])
                print('Предсказано:', [app_to_title[a] for a in self.apps[pred_items]])
                print('Правильно:', [app_to_title[a] for a in self.apps[true_items]])
            current_precision = self.precision(pred_items, true_items, k)
            current_ndcg = self.ndcg_at_k(pred_items, true_items, k)
            total_precision += current_precision
            total_ndcg += current_ndcg
            total_els += 1
            if debug:
                print(f'precision = {current_precision}, ndcg = {current_ndcg}')
        return {'precision': total_precision / max(total_els,1), 'ndcg': total_ndcg / max(total_els,1)}

#users, apps, u_idx, i_idx = get_new_indexes(train)
#R_train = get_csr_matrix(train, users, apps, u_idx, i_idx)
#R_val   = get_csr_matrix(val,   users, apps, u_idx, i_idx)
#R_test  = get_csr_matrix(test,  users, apps, u_idx, i_idx)
