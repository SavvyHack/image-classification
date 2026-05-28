# -----------------------------
# Import required libraries
# -----------------------------

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Remove unnecessary warning messages
# -----------------------------

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------
# Set folder paths
# -----------------------------

BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "task2_data"
OUT_DIR    = BASE_DIR / "outputs"
FIG_DIR    = BASE_DIR / "figures"
CNN_PATH   = OUT_DIR / "cnn_features_task2.csv"
EXTRA_PATH = OUT_DIR / "extra_features_task2.csv"


# -----------------------------
# Set reproducibility and validation settings
# -----------------------------

RANDOM_STATE = 0
N_SPLITS     = 5
MAIN_MODELS  = ("logreg", "svm_rbf", "rf")


# -----------------------------
# Define provided feature files
# -----------------------------

PROVIDED_FILES = {"color":      "color_histogram.csv",
                  "hog":        "hog_pca.csv",
                  "additional": "additional_features.csv"}


def load_data():
    """Load metadata and available feature files for Task 2."""

    # Load train and test metadata
    tr = pd.read_csv(DATA_DIR / "train_metadata.csv")
    te = pd.read_csv(DATA_DIR / "test_metadata.csv")

    # Load the provided feature files into a dictionary
    feats = {n: pd.read_csv(DATA_DIR / f) for n, f in PROVIDED_FILES.items()}

    # Add extra handcrafted features and CNN features if they exist
    for name, path in (("extra", EXTRA_PATH), ("cnn", CNN_PATH)):
        if path.exists():
            feats[name] = pd.read_csv(path)
            print(f"  {name} features: {feats[name].shape[1] - 1} dims")
        else:
            print(f"  {name} features missing -- run extract_{name}_features_task2.py")

    return tr, te, feats


def build_matrix(meta, feats, groups):
    """Concatenate selected feature groups in metadata image_id order."""

    # Join each selected feature group by image_id and combine them into one matrix
    return np.hstack([feats[g].set_index("image_id").loc[meta["image_id"]].values
                      for g in groups])


def make_feature_sets(feats):
    """Create feature-set combinations depending on which files are available."""

    # Start with the original features provided in the assignment
    fs = {"provided": ["color", "hog", "additional"]}

    # Add handcrafted features if available
    if "extra" in feats:
        fs["provided+extra"] = fs["provided"] + ["extra"]

    # Add CNN-based feature combinations if available
    if "cnn" in feats:
        fs["cnn_only"]     = ["cnn"]
        fs["provided+cnn"] = fs["provided"] + ["cnn"]

        # Use all feature groups if both extra and CNN features exist
        if "extra" in feats:
            fs["all"] = fs["provided"] + ["extra", "cnn"]

    return fs


def _pipe(clf):
    """Create a model pipeline with standardisation followed by a classifier."""

    # Scale features before fitting the classifier
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def model_grids():
    """Return model pipelines and parameter grids for tuning."""

    # Define the real models and the hyperparameters to test
    return {
        "logreg": (_pipe(LogisticRegression(max_iter=2000,
                                            random_state=RANDOM_STATE)),
                   {"clf__C": [0.01, 0.1, 1, 10]}),
        "svm_rbf": (_pipe(SVC(kernel="rbf", random_state=RANDOM_STATE)),
                    {"clf__C":     [0.1, 1, 10],
                     "clf__gamma": ["scale", 0.001, 0.01]}),
        "rf": (_pipe(RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE)),
               {"clf__n_estimators": [300, 500],
                "clf__max_depth":    [None, 20],
                "clf__max_features": ["sqrt", "log2"]}),
    }


def grid_search_eval(X, y, est, grid, cv):
    """Tune a model and return cross-validation accuracy and macro-F1."""

    # Search over the parameter grid using cross-validation
    gs = GridSearchCV(
        est, grid, cv=cv, n_jobs=-1, refit="accuracy",
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
    )

    # Fit the grid search on the training data
    gs.fit(X, y)

    # Extract the best result from the grid search
    i = gs.best_index_

    return {
        "accuracy":    float(gs.cv_results_["mean_test_accuracy"][i]),
        "macro_f1":    float(gs.cv_results_["mean_test_macro_f1"][i]),
        "best_params": gs.best_params_,
        "estimator":   gs.best_estimator_,
    }


def dummy_eval(X, y, cv):
    """Evaluate a majority-class baseline model."""

    # Create a simple baseline that always predicts the most common class
    clf = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)

    # Generate cross-validated predictions for the baseline
    y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

    return {"accuracy":    accuracy_score(y, y_pred),
            "macro_f1":    f1_score(y, y_pred, average="macro"),
            "best_params": {}}


def plot_confusion(cm, labels, title, savepath):
    """Save a row-normalised confusion matrix."""

    # Convert raw counts into row-normalised proportions
    cm_n = cm / cm.sum(axis=1, keepdims=True)

    # Create the confusion matrix figure
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_n, vmin=0, vmax=1)

    # Add class labels to both axes
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)

    # Label the figure
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)

    # Write the numeric values inside each confusion matrix cell
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_n[i, j]:.2f}",
                    ha="center", va="center", fontsize=8)

    # Save the figure to disk
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(savepath, dpi=150); plt.close()


def plot_accuracy_bars(df, savepath):
    """Save a bar chart comparing CV accuracy for all runs."""

    # Create combined labels for each feature-set and model combination
    df = df.copy()
    df["combo"] = df["features"] + " + " + df["model"]
    df = df.sort_values("accuracy", ascending=False)

    # Create the accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(df)), df["accuracy"])

    # Label the chart
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["combo"], rotation=45, ha="right")
    ax.set_ylabel("Cross-validation accuracy")
    ax.set_title("Task 2 model and feature comparison")

    # Save the figure to disk
    plt.tight_layout(); plt.savefig(savepath, dpi=150); plt.close()


def save_clf_report(y_true, y_pred, names, row, savepath):
    """Save a text classification report for one model."""

    # Save model details and the full classification report
    with open(savepath, "w", encoding="utf-8") as f:
        f.write(f"Features: {row['features']}\n"
                f"Model: {row['model']}\n"
                f"Accuracy: {row['accuracy']:.4f}\n"
                f"Macro-F1: {row['macro_f1']:.4f}\n"
                f"Best parameters: {row['best_params']}\n\n")
        f.write(classification_report(y_true, y_pred,
                                      target_names=names, digits=3))


def top_confused_pairs(cm, names, k=10):
    """Return the most common off-diagonal confusion pairs."""

    # Store all cases where the true class and predicted class are different
    pairs = [{"true":             names[i],
              "predicted":        names[j],
              "count":            int(cm[i, j]),
              "recall_lost_frac": cm[i, j] / max(cm[i].sum(), 1)}
             for i in range(len(names)) for j in range(len(names))
             if i != j and cm[i, j] > 0]

    # Return the most common mistakes
    return pd.DataFrame(sorted(pairs, key=lambda p: p["count"], reverse=True)[:k])


def main():
    """Run the full Task 2 model comparison and final prediction pipeline."""

    # Create output folders if they do not already exist
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    # Load metadata and feature files
    tr_meta, te_meta, feats = load_data()

    # Extract labels and class names
    y     = tr_meta["class_id"].values
    names = (tr_meta.drop_duplicates("class_id")
                    .sort_values("class_id")["class_name"].tolist())

    # Print basic dataset information
    print(f"\nClasses ({len(names)}): {names}")
    print(f"Train: {len(tr_meta)}  Test: {len(te_meta)}\n")

    # Create feature-set combinations, model grids, and cross-validation folds
    feature_sets = make_feature_sets(feats)
    grids        = model_grids()
    cv           = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                   random_state=RANDOM_STATE)

    # Store summary results and fitted estimators for later analysis
    results = []
    fitted  = {}

    # Train every real model on every feature set
    for feat, groups in feature_sets.items():
        X = build_matrix(tr_meta, feats, groups)
        print(f"Feature set: {feat:18s} shape={X.shape}")

        for m_name, (est, grid) in grids.items():
            t0 = time.time()

            # Tune model and record cross-validation scores
            r  = grid_search_eval(X, y, est, grid, cv)

            print(f"  {m_name:18s} acc={r['accuracy']:.4f} "
                  f"macroF1={r['macro_f1']:.4f}  ({time.time()-t0:.1f}s) "
                  f"best={r['best_params']}")

            # Save this run's summary results
            results.append({"features":    feat,
                            "model":       m_name,
                            "accuracy":    r["accuracy"],
                            "macro_f1":    r["macro_f1"],
                            "best_params": json.dumps(r["best_params"])})

            # Save the best fitted estimator for later out-of-fold predictions
            fitted[(feat, m_name)] = r["estimator"]

        # Evaluate and save the dummy baseline for the same feature set
        d = dummy_eval(X, y, cv)
        print(f"  {'dummy':18s} acc={d['accuracy']:.4f}")
        results.append({"features":    feat,
                        "model":       "dummy_most_frequent",
                        "accuracy":    d["accuracy"],
                        "macro_f1":    d["macro_f1"],
                        "best_params": json.dumps({})})
        print()

    # Convert results into a sorted dataframe
    res_df = (pd.DataFrame(results)
                .sort_values("accuracy", ascending=False)
                .reset_index(drop=True))

    # Save all cross-validation results and the accuracy comparison chart
    res_df.to_csv(OUT_DIR / "cv_results_task2.csv", index=False)
    plot_accuracy_bars(res_df, FIG_DIR / "accuracy_comparison_task2.png")

    print("\nCross-validation results:")
    print(res_df.to_string(index=False))

    def oof_predict(feat, m_name):
        """Generate out-of-fold predictions for a selected feature/model combination."""

        # Build the matching feature matrix and generate cross-validated predictions
        X = build_matrix(tr_meta, feats, feature_sets[feat])
        return cross_val_predict(fitted[(feat, m_name)], X, y,
                                 cv=cv, n_jobs=-1)

    # Save reports and confusion matrices for each main model
    for m_name in MAIN_MODELS:
        sub = res_df[res_df["model"] == m_name]

        if sub.empty:
            continue

        # Select this model's best-performing feature set
        row    = sub.iloc[0]

        # Generate out-of-fold predictions only for the model being analysed
        y_pred = oof_predict(row["features"], m_name)

        # Save confusion matrix and classification report
        plot_confusion(confusion_matrix(y, y_pred), names,
                       title=f"{m_name}: {row['features']}",
                       savepath=FIG_DIR / f"confusion_{m_name}_task2.png")
        save_clf_report(y, y_pred, names, row,
                        OUT_DIR / f"classification_report_{m_name}_task2.txt")

    # Select the best overall model and feature set
    best = res_df.iloc[0]

    print("\nBest overall configuration:")
    print(best)

    # Save the headline confusion matrix, report, and confused-pairs table
    if best["model"] in MAIN_MODELS:
        y_pred_best = oof_predict(best["features"], best["model"])
        cm_best     = confusion_matrix(y, y_pred_best)

        plot_confusion(cm_best, names,
                       title=f"Best: {best['features']} + {best['model']}",
                       savepath=FIG_DIR / "confusion_matrix_best_task2.png")
        save_clf_report(y, y_pred_best, names, best,
                        OUT_DIR / "classification_report_best_task2.txt")

        # Save the most common confused class pairs
        pairs_df = top_confused_pairs(cm_best, names, k=10)
        pairs_df.to_csv(OUT_DIR / "top_confused_pairs_task2.csv", index=False)

        print("\nTop confused pairs for best configuration:")
        print(pairs_df.to_string(index=False))

    # Build final train and test matrices using the best feature set
    groups  = feature_sets[best["features"]]
    X_train = build_matrix(tr_meta, feats, groups)
    X_test  = build_matrix(te_meta, feats, groups)

    # Recreate the best model and apply the best hyperparameters
    if best["model"] == "dummy_most_frequent":
        final = DummyClassifier(strategy="most_frequent",
                                random_state=RANDOM_STATE)
    else:
        final = grids[best["model"]][0]
        final.set_params(**json.loads(best["best_params"]))

    # Train on the full labelled training set
    final.fit(X_train, y)

    # Save predictions in submission format
    pd.DataFrame({"image_id": te_meta["image_id"].values,
                  "class_id": final.predict(X_test)}
                 ).to_csv(OUT_DIR / "task2_submission.csv", index=False)

    print(f"\nSaved final submission: {OUT_DIR / 'task2_submission.csv'}")

# -----------------------------
    # Save per-model Kaggle submissions
    # -----------------------------

    # For each of the three main models, train on its own best feature set
    # using its tuned hyperparameters and write a separate Kaggle submission.
    # Naming pattern: task2_submission_<modelname>.csv
    for m_name in MAIN_MODELS:
        sub = res_df[res_df["model"] == m_name]

        if sub.empty:
            continue

        # This model's best row (res_df is sorted by accuracy descending, so
        # iloc[0] of the filtered subset is this model's top-scoring run)
        m_row    = sub.iloc[0]
        m_groups = feature_sets[m_row["features"]]

        # Build train and test matrices using this model's best feature set
        X_train_m = build_matrix(tr_meta, feats, m_groups)
        X_test_m  = build_matrix(te_meta, feats, m_groups)

        # Recreate this model's pipeline and apply its tuned hyperparameters
        # (best_params is stored as a JSON string in the results dataframe)
        model_m = grids[m_name][0]
        model_m.set_params(**json.loads(m_row["best_params"]))
        model_m.fit(X_train_m, y)

        # Save predictions in submission format
        sub_path = OUT_DIR / f"task2_submission_{m_name}.csv"
        pd.DataFrame({"image_id": te_meta["image_id"].values,
                      "class_id": model_m.predict(X_test_m)}
                     ).to_csv(sub_path, index=False)

        print(f"Saved {m_name} submission: {sub_path}")

# -----------------------------
# Run the script
# -----------------------------

if __name__ == "__main__":
    main()