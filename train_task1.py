"""
train_task1.py
--------------
Train and evaluate models on Task 1 (coarse-grained animal classification).

Pipeline:
  1. Load provided features (color hist, HOG-PCA, additional) plus, if
     present, CNN features extracted by extract_cnn_features.py.
  2. Standardise each feature group.
  3. 5-fold stratified cross-validation across multiple (feature, model)
     combinations -- reporting accuracy and macro-F1.
  4. Refit the best (feature, model) combination on all training data and
     produce a Kaggle submission CSV.
  5. Save a confusion matrix figure for the best config.

Models used (3 distinct algorithms, per the solo-project spec):
  - Logistic Regression (multinomial, linear)
  - Support Vector Machine, RBF kernel (non-linear, kernel-based)
  - Random Forest (ensemble of decision trees)

These three were chosen because they sit in different model families with
different inductive biases -- enabling a substantive comparison in the
report rather than tweaking hyperparameters of a single algorithm.
"""

import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing  import StandardScaler
from sklearn.linear_model   import LogisticRegression
from sklearn.svm            import SVC
from sklearn.ensemble       import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics        import (accuracy_score, f1_score,
                                    confusion_matrix, classification_report)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = "task1_data"
OUT_DIR  = "outputs"
FIG_DIR  = "figures"
CNN_PATH   = os.path.join(OUT_DIR, "cnn_features.csv")
EXTRA_PATH = os.path.join(OUT_DIR, "extra_features.csv")

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_data():
    tr_meta = pd.read_csv(os.path.join(DATA_DIR, "train_metadata.csv"))
    te_meta = pd.read_csv(os.path.join(DATA_DIR, "test_metadata.csv"))

    feat_files = {
        "color":      "color_histogram.csv",
        "hog":        "hog_pca.csv",
        "additional": "additional_features.csv",
    }
    feats = {name: pd.read_csv(os.path.join(DATA_DIR, f))
             for name, f in feat_files.items()}

    if os.path.exists(EXTRA_PATH):
        feats["extra"] = pd.read_csv(EXTRA_PATH)
        print(f"  Extra features found ({feats['extra'].shape[1]-1} dims) -- "
              "will be included.")

    if os.path.exists(CNN_PATH):
        feats["cnn"] = pd.read_csv(CNN_PATH)
        print(f"  CNN features found ({feats['cnn'].shape[1]-1} dims) -- "
              "will be included.")
    else:
        print("  CNN features not found -- running without them.\n"
              f"  (To include them, run extract_cnn_features.py then re-run this.)")

    return tr_meta, te_meta, feats


def build_matrices(meta, feats, feature_set):
    """Concatenate the requested feature groups, joining by image_id, in the
    order given by `meta['image_id']`. Returns an (N, D) numpy array."""
    parts = []
    for name in feature_set:
        df = feats[name].set_index("image_id").loc[meta["image_id"]]
        parts.append(df.values)
    return np.hstack(parts)


# ---------------------------------------------------------------------------
# 2. Model definitions
# ---------------------------------------------------------------------------
def make_models():
    return {
        "logreg":  LogisticRegression(max_iter=2000, C=1.0, random_state=0),
        "svm_rbf": SVC(kernel="rbf", C=5.0, gamma="scale", random_state=0),
        "rf":      RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                          random_state=0),
    }


# ---------------------------------------------------------------------------
# 3. Cross-validation evaluation
# ---------------------------------------------------------------------------
def cv_evaluate(X, y, model_name, model, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    # Standardise inside the CV via cross_val_predict on a pipeline
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf", model)])
    y_pred = cross_val_predict(pipe, X, y, cv=skf, n_jobs=-1)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "macro_f1": f1_score(y, y_pred, average="macro"),
        "y_pred":   y_pred,
    }


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    tr_meta, te_meta, feats = load_data()
    y_train = tr_meta["class_id"].values
    class_names = (tr_meta.drop_duplicates("class_id")
                          .sort_values("class_id")["class_name"].tolist())

    # Feature subsets to compare. We always run "provided" as a baseline,
    # and conditionally add subsets that use the extra hand-crafted features
    # (LBP + HSV) and the CNN embeddings if they were extracted.
    feature_sets = {
        "provided": ["color", "hog", "additional"],
    }
    if "extra" in feats:
        feature_sets["provided+extra"] = ["color", "hog", "additional", "extra"]
    if "cnn" in feats:
        feature_sets["cnn_only"]      = ["cnn"]
        feature_sets["provided+cnn"]  = ["color", "hog", "additional", "cnn"]
        if "extra" in feats:
            feature_sets["all"] = ["color", "hog", "additional", "extra", "cnn"]

    models = make_models()

    print(f"\nClasses ({len(class_names)}): {class_names}")
    print(f"Train: {len(y_train)}   Test: {len(te_meta)}\n")

    results = []
    fold_preds = {}     # keep best fold predictions for confusion matrix

    for feat_name, subset in feature_sets.items():
        X = build_matrices(tr_meta, feats, subset)
        print(f"Feature set '{feat_name}' shape: {X.shape}")
        for m_name, m in models.items():
            t0 = time.time()
            r = cv_evaluate(X, y_train, m_name, m)
            print(f"  {m_name:10s}  acc={r['accuracy']:.4f}  "
                  f"macroF1={r['macro_f1']:.4f}  ({time.time()-t0:.1f}s)")
            results.append({
                "features": feat_name,
                "model":    m_name,
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
            })
            fold_preds[(feat_name, m_name)] = r["y_pred"]
        print()

    # ---- Save results table
    res_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    res_df.to_csv(os.path.join(OUT_DIR, "cv_results.csv"), index=False)
    print("CV results (sorted by accuracy):")
    print(res_df.to_string(index=False))

    # ---- Pick best config, plot confusion matrix
    best = res_df.iloc[0]
    best_subset = feature_sets[best["features"]]
    print(f"\nBest config: features={best['features']}, model={best['model']}")

    y_pred_best = fold_preds[(best["features"], best["model"])]
    cm = confusion_matrix(y_train, y_pred_best)
    plot_confusion(cm, class_names,
                   title=f"CV confusion matrix -- {best['features']} + {best['model']}",
                   savepath=os.path.join(FIG_DIR, "confusion_matrix_best.png"))

    # Also save the full classification report
    rpt = classification_report(y_train, y_pred_best,
                                target_names=class_names, digits=3)
    with open(os.path.join(OUT_DIR, "classification_report_best.txt"), "w") as f:
        f.write(f"Best config: features={best['features']}, model={best['model']}\n")
        f.write(f"CV accuracy: {best['accuracy']:.4f}\n")
        f.write(f"CV macro-F1: {best['macro_f1']:.4f}\n\n")
        f.write(rpt)
    print(rpt)

    # ---- Refit on full training data, predict on test, write Kaggle submission
    print("Refitting on full training data and predicting on test set...")
    X_train_full = build_matrices(tr_meta, feats, best_subset)
    X_test       = build_matrices(te_meta, feats, best_subset)

    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf",   make_models()[best["model"]])])
    pipe.fit(X_train_full, y_train)
    y_test_pred = pipe.predict(X_test)

    submission = pd.DataFrame({
        "image_id": te_meta["image_id"].values,
        "class_id": y_test_pred,
    })
    submission_path = os.path.join(OUT_DIR, "task1_submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Wrote {submission_path}  ({len(submission)} predictions)")


def plot_confusion(cm, labels, title, savepath):
    """Matplotlib confusion matrix. Row-normalised."""
    cm_n = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_n[i,j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_n[i,j] > 0.5 else "black",
                    fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()
    print(f"Saved {savepath}")


if __name__ == "__main__":
    main()
