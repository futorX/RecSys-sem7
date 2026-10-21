# ФИО

Сметанин Илья Александрович, Б05-306

# Датасет

Название **Game Recommendations on Steam**

Ссылка на источник https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam/data

# Инструкции загрузки данных и запуска

Рекоммендую запускать из Colab. Для этого откройте в Colab файл experiments_and_models.ipynb, запустите раздел ячеек 0.2 для загрузки данных, 0.3 для записи файлов моделей (ничего устанавливать не надо)

Все модели имеют похожий интерфейс. Запускать модели можно так:

```
estim = UserBasedCollaborativeFiltering(train)
estim.train()
metrics = estim.test(val, regularization=regularization, k_neighbors=kn)
```

```
estim = ALS(train)
estim.train(d=d, epoch_num=epoch_num, alpha=alpha, lambd=lambd, gamma=gamma)
metrics = estim.test(val, cos_sim=False)
```

```
estim = EASE(train)
estim.train(lambd=lambd)
metrics_test = estim.test(val)
```

```
fm_estim = FM(train, games_df, users_df, metadata_path=f"{path}/games_metadata.json")
fm_estim.train(n_factors=n_factors, epochs=epochs, lr=lr, reg=reg, top_tags=top_tags, batch_size=4096, descr_emd_dim=descr_emd_dim)
metrics = fm_estim.test(val)
```

```
estim = IALS(train)
estim.train(d=d, epoch_num=epoch_num, alpha=alpha, lambd=lambd, gamma=gamma, neg_weight=neg_weight, hours_pos_w=hours_pos_w, hours_neg_w=hours_neg_w)
metrics = estim.test(val, cos_sim=False)
```