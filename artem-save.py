import pandas as pd
import numpy as np

X_tr_feat.to_parquet("X_tr_feat_clip.parquet")
X_te_feat.to_parquet("X_te_feat_clip.parquet")
np.save("y.npy", y)

#добавить в новый ноутбук

import pandas as pd
import numpy as np

X_tr_feat = pd.read_parquet("X_tr_feat_clip.parquet")
X_te_feat = pd.read_parquet("X_te_feat_clip.parquet")
y = np.load("y.npy")

def add_custom_features_post(X_tr: pd.DataFrame, X_te: pd.DataFrame):
    X_tr = X_tr.copy()
    X_te = X_te.copy()

    # 🔽 Примеры, сюда пишешь свою магию:
    # Пример: квадраты нескольких колонок
    important_cols = [c for c in X_tr.columns if "clip_emb_" in c][:50]  # первые 50 CLIP-фичей, например
    for col in important_cols:
        X_tr[col + "_sq"] = X_tr[col] ** 2
        X_te[col + "_sq"] = X_te[col] ** 2

    # Пример: сумма/норма CLIP-вектора (может помочь)
    clip_cols = [c for c in X_tr.columns if c.startswith("clip_emb_")]
    if clip_cols:
        X_tr["clip_sum"] = X_tr[clip_cols].sum(axis=1)
        X_te["clip_sum"] = X_te[clip_cols].sum(axis=1)

    return X_tr, X_te


from catboost import CatBoostRegressor, Pool

train_pool = Pool(X_tr_feat, y)
test_pool = Pool(X_te_feat)

model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=4000,
    depth=8,
    learning_rate=0.03,
    task_type="GPU",
    eval_metric="RMSE",
)
model.fit(train_pool, verbose=200)

pred = model.predict(test_pool)
