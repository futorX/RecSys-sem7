
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

class ALS(Estimator):
    def __init__(self, train_df):
        super().__init__(train_df)
    def train(self, d=10, epoch_num=10, alpha=10.0, lambd=0.5, gamma=0.1):
        self.R_train = self.get_csr_matrix(self.train_df)
        user_num, item_num = self.R_train.shape
        R_csc = self.R_train.tocsc()

        rng = np.random.default_rng(0)
        self.X = rng.normal(0, 0.1, (d, user_num)) # d * u
        self.Y = rng.normal(0, 0.1, (d, item_num)) # d * i
        C_i = np.zeros(user_num)
        C_j = np.zeros(item_num)
        denom = 0
        for user in range(user_num):
            temp = float(self.R_train[user].nnz) ** gamma
            denom += temp
            C_i[user] = temp
        if denom != 0:
            C_i = C_i * user_num / denom
        else:
            C_i = np.ones(user_num)
        denom = 0
        for item in range(item_num):
            temp = float(R_csc[:,item].nnz) ** gamma
            denom += temp
            C_j[item] = temp
        if denom != 0:
            C_j = C_j * item_num / denom
        else:
            C_j = np.ones(item_num)
        eye = np.eye(d)
        for _ in range(epoch_num):
            YYT = self.Y @ self.Y.T
            for u in range(user_num):
                idx = self.R_train[u].indices
                if len(idx) == 0: continue
                Y_u = self.Y[:, idx]
                A = YYT + alpha * (Y_u @ Y_u.T) + lambd * eye * C_i[u]
                b = (1.0 + alpha) * Y_u.sum(axis=1)
                self.X[:, u] = np.linalg.solve(A, b) # A^{-1} b
            XXT = self.X @ self.X.T
            for i in range(item_num):
                idx = R_csc[:, i].indices
                if len(idx) == 0: continue
                X_i = self.X[:, idx]
                A = XXT + alpha * (X_i @ X_i.T) + lambd * eye * C_j[i]
                b = (1.0 + alpha) * X_i.sum(axis=1)
                self.Y[:, i] = np.linalg.solve(A, b) # A^{-1} b

    def get_prediction(self, ind, k=10, cos_sim=True):
        user_num, item_num = self.R_train.shape
        user_embedding = self.X[:, ind]

        dots = user_embedding @ self.Y
        # dots[i] - это <user_embedding, item_i_embedding>
        if cos_sim:
            user_norm = np.linalg.norm(user_embedding)
            item_norms = np.linalg.norm(self.Y, axis=0)
            # item_norms[i] - это норма item_i
            scores = dots / (user_norm * item_norms)
        else:
            scores = dots
        scores[~np.isfinite(scores)] = 0.0
        seen = self.R_train[ind].indices
        scores[seen] = -np.inf

        order = np.argsort(-scores)
        order = order[np.isfinite(scores[order])][:k]
        return scores, order
    def test(self, value_df, k=10, debug=False, app_to_title=lambda x:x, cos_sim=True):
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
            pred_scores, pred_items = self.get_prediction(ind, k, cos_sim=cos_sim)
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
