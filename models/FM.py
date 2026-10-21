
from Estimator import Estimator

# Код с семинара
import numpy as np
import scipy.sparse as sparse
import pandas as pd
from tqdm import tqdm
import json
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch
import kagglehub
import matplotlib.pyplot as plt
from datetime import datetime
import os, shutil, json
import gdown
import sklearn
import itertools

def sample_negatives(rng, user_indices, seen_mask, candidate_ids):
    if np.any(seen_mask[user_indices].all(axis=1)):
        raise ValueError("A sampled user has seen every candidate")
    
    negative = rng.choice(candidate_ids, size=len(user_indices))
    collisions = seen_mask[user_indices, negative]
    
    while np.any(collisions):
        negative[collisions] = rng.choice(candidate_ids, size=int(collisions.sum()))
        collisions = seen_mask[user_indices, negative]
        
    return negative

class FactorizationMachine:
    def __init__(self, n_factors=16, epochs=30, lr=0.01, reg=1e-5,
                 batch_size=512, random_state=42, progress=True):
        self.n_factors, self.epochs, self.lr = n_factors, epochs, lr
        self.reg, self.batch_size, self.random_state = reg, batch_size, random_state
        self.progress = progress

    def initialize(self, features):
        self.features = features
        rng = np.random.default_rng(self.random_state)
        self.w = np.zeros(features.n_features)
        self.V = 0.05 * rng.standard_normal((features.n_features, self.n_factors))
        self.m_w, self.s_w = np.zeros_like(self.w), np.zeros_like(self.w)
        self.m_V, self.s_V = np.zeros_like(self.V), np.zeros_like(self.V)
        self.step = 0

    def forward(self, X):
        squared = X.multiply(X)
        projected = X @ self.V
        scores = np.asarray(X @ self.w).ravel() + 0.5 * np.sum(
            projected ** 2 - squared @ (self.V ** 2), axis=1
        )
        return scores, projected, squared

    def gradients(self, X, projected, squared, errors):
        grad_w = np.asarray(X.T @ errors).ravel()
        grad_V = (X.T @ (errors[:, None] * projected)
                  - np.asarray(squared.T @ errors).ravel()[:, None] * self.V)
        return grad_w, np.asarray(grad_V)

    def bpr_gradients(self, X_pos, X_neg):
        # BPR лосс (Bayesian Personalized Ranking)
        pos, projected_pos, squared_pos = self.forward(X_pos)
        neg, projected_neg, squared_neg = self.forward(X_neg)
        delta = pos - neg
        weight = np.exp(-delta) / (1 + np.exp(-delta))
        
        gw_pos, gv_pos = self.gradients(X_pos, projected_pos, squared_pos, -weight)
        gw_neg, gv_neg = self.gradients(X_neg, projected_neg, squared_neg, weight)
        size = X_pos.shape[0]
        loss = float(np.logaddexp(0, -delta).mean())
        loss += 0.5 * self.reg * (np.sum(self.w ** 2) + np.sum(self.V ** 2))
        
        return loss, (gw_pos + gw_neg) / size + self.reg * self.w, \
               (gv_pos + gv_neg) / size + self.reg * self.V

    def update(self, grad_w, grad_V):
        self.step += 1
        for parameter, first, second, gradient in [
            (self.w, self.m_w, self.s_w, grad_w),
            (self.V, self.m_V, self.s_V, grad_V),
        ]:
            first *= 0.9
            first += 0.1 * gradient
            second *= 0.999
            second += 0.001 * gradient ** 2
            parameter -= self.lr * (first / (1 - 0.9 ** self.step)) / (
                np.sqrt(second / (1 - 0.999 ** self.step)) + 1e-8
            )

    def score_matrix(self):
        boundary = self.features.user_dim
        U, I = self.features.user_features, self.features.item_features
        Vu, Vi = self.V[:boundary], self.V[boundary:]
        projected_u, projected_i = U @ Vu, I @ Vi
        
        offset_u = U @ self.w[:boundary] + 0.5 * np.sum(
            projected_u ** 2 - U.multiply(U) @ (Vu ** 2), axis=1
        )
        offset_i = I @ self.w[boundary:] + 0.5 * np.sum(
            projected_i ** 2 - I.multiply(I) @ (Vi ** 2), axis=1
        )
        
        return offset_u[:, None] + offset_i[None, :] + projected_u @ projected_i.T

    def fit(self, history, features):
        self.initialize(features)
        positives = history.drop_duplicates(["user_idx", "item_idx"])
        
        seen = sparse.csr_matrix(
            (np.ones(len(history)), (history.user_idx, history.item_idx)),
            shape=(len(features.users), len(features.apps))
        ).toarray().astype(bool)
        
        candidates = np.arange(len(features.apps))
        positive_users = positives.user_idx.to_numpy()
        positive_items = positives.item_idx.to_numpy()
        
        rng = np.random.default_rng(self.random_state)
        
        with tqdm(total=self.epochs, desc="FM-BPR Training", disable=True) as pbar:
            for epoch in range(1, self.epochs + 1):
                order = rng.permutation(len(positives))
                total_loss = 0.0
                
                for start in range(0, len(order), self.batch_size):
                    batch = order[start:start + self.batch_size]
                    u, i = positive_users[batch], positive_items[batch]
                    j = sample_negatives(rng, u, seen, candidates)
                    
                    X_pos, X_neg = features.pairs(u, i), features.pairs(u, j)
                    loss, grad_w, grad_V = self.bpr_gradients(X_pos, X_neg)
                    
                    self.update(grad_w, grad_V)
                    total_loss += loss * len(batch)
                    
                pbar.set_postfix({"loss": f"{total_loss / len(order):.4f}"})
                pbar.update(1)
        return self

# Свои данные

class FeatureMap:
    def __init__(self, train_df, games_df, users_df, metadata_path, top_tags=20, descr_emd_dim=16):
        self.users = np.sort(train_df["user_id"].unique())
        self.apps  = np.sort(train_df["app_id"].unique())
        self.u_idx = {u: i for i, u in enumerate(self.users)}
        self.i_idx = {a: i for i, a in enumerate(self.apps)}
        
        u_feats = users_df.set_index("user_id").reindex(self.users).fillna(0)
        u_side = np.log1p(u_feats[["products", "reviews"]].values)
        u_side = (u_side - u_side.mean(axis=0)) / (u_side.std(axis=0) + 1e-9)
        
        g_feats = games_df.set_index("app_id").reindex(self.apps).fillna(0)
        g_feats["release_year"] = pd.to_datetime(g_feats["date_release"], errors='coerce').dt.year.fillna(2000)
        g_feats["log_user_reviews"] = np.log1p(g_feats["user_reviews"])
        
        i_side = g_feats[["price_final", "positive_ratio", "release_year", "log_user_reviews"]].values
        i_side = (i_side - i_side.mean(axis=0)) / (i_side.std(axis=0) + 1e-9)
        
        with open(metadata_path, 'r') as f:
            metadata = [json.loads(line) for line in f]
        meta_dict = {item['app_id']: item for item in metadata}
        
        tag_matrix = np.zeros((len(self.apps), 0), dtype=np.float32)
        
        if top_tags and isinstance(top_tags, int) and top_tags > 0:
            tag_counter = Counter()
            for app_id in self.apps:
                if app_id in meta_dict:
                    tag_counter.update(meta_dict[app_id].get('tags', []))
                    
            top_tags_list = [tag for tag, _ in tag_counter.most_common(top_tags)]
            tag_to_idx = {tag: i for i, tag in enumerate(top_tags_list)}
            
            tag_matrix = np.zeros((len(self.apps), len(top_tags_list)), dtype=np.float32)
            for i, app_id in enumerate(self.apps):
                if app_id in meta_dict:
                    for tag in meta_dict[app_id].get('tags', []):
                        if tag in tag_to_idx:
                            tag_matrix[i, tag_to_idx[tag]] = 1.0

        descriptions = [meta_dict[app_id].get('description', '') if app_id in meta_dict else '' for app_id in self.apps]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        embeddings = encoder.encode(descriptions, batch_size=512, show_progress_bar=True)
        
        pca = PCA(n_components=descr_emd_dim, random_state=0)
        text_emb = pca.fit_transform(embeddings)
        text_emb = (text_emb - text_emb.mean(axis=0)) / (text_emb.std(axis=0) + 1e-9)
        
        i_side = np.hstack([i_side, tag_matrix, text_emb])
        
        u_ids = sparse.eye(len(self.users), format="csr")
        i_ids = sparse.eye(len(self.apps), format="csr")
        
        self.user_features = sparse.hstack([u_ids, sparse.csr_matrix(u_side)], format="csr")
        self.item_features = sparse.hstack([i_ids, sparse.csr_matrix(i_side)], format="csr")
        
        self.user_dim = self.user_features.shape[1]
        self.n_features = self.user_dim + self.item_features.shape[1]
        print(f"Итоговое число признаков: {self.n_features} (включая {top_tags} тегов и {descr_emd_dim} эмбеддингов)")
        
    def pairs(self, user_indices, item_indices):
        return sparse.hstack([
            self.user_features[user_indices], 
            self.item_features[item_indices]
        ], format="csr")


class FM(Estimator):
    def __init__(self, train_df, games_df, users_df, metadata_path):
        super().__init__(train_df)
        self.games_df = games_df
        self.users_df = users_df
        self.metadata_path = metadata_path
        self.model = None
        self.feature_map = None
        self.cached_scores = None

    def train(self, n_factors=16, epochs=10, lr=0.01, reg=1e-5, batch_size=1024, top_tags=50, descr_emd_dim=16):
        self.R_train = self.get_csr_matrix(self.train_df)
        
        self.feature_map = FeatureMap(self.train_df, self.games_df, self.users_df, self.metadata_path, top_tags, descr_emd_dim)
        
        pos_train = self.train_df[self.train_df["is_recommended"] == True].copy()
        pos_train["user_idx"] = pos_train["user_id"].map(self.feature_map.u_idx).astype(int)
        pos_train["item_idx"] = pos_train["app_id"].map(self.feature_map.i_idx).astype(int)
        
        self.model = FactorizationMachine(
            n_factors=n_factors, epochs=epochs, lr=lr, 
            reg=reg, batch_size=batch_size, random_state=0
        )
        self.model.fit(pos_train, self.feature_map)
        self.cached_scores = self.model.score_matrix()

    def get_prediction(self, ind, k=10):
        scores = self.cached_scores[ind].copy()
        seen = self.R_train[ind].indices
        scores[seen] = -np.inf

        order = np.argsort(-scores)
        order = order[np.isfinite(scores[order])][:k]
        return scores, order

    def test(self, value_df, k=10, debug=False, app_to_title=lambda x:x):
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
            
            pred_scores, pred_items = self.get_prediction(ind, k)
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
