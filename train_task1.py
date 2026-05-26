"""
Train and evaluate Task 1 classifiers.

This script compares multiple feature representations and learning algorithms,
tunes hyperparameters with cross-validation, saves model comparison results,
and generates figures for error analysis.
"""

# -----------------------------
# Import required libraries
# -----------------------------

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# -----------------------------
# Remove unnecessary warning messages
# -----------------------------

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------
# Set folder paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "task1_data"
OUT_DIR = BASE_DIR / "outputs"
FIG_DIR = BASE_DIR / "figures"
MISCLASS_DIR = FIG_DIR / "misclassified_examples"

CNN_PATH = OUT_DIR / "cnn_features.csv"
EXTRA_PATH = OUT_DIR / "extra_features.csv"


# -----------------------------
# Set reproducibility and validation settings
# -----------------------------

RANDOM_STATE = 0
N_SPLITS = 5


def load_data():
    """Load metadata and available feature files for Task 1."""

    # Load train and test metadata
    tr_meta = pd.read_csv(DATA_DIR / "train_metadata.csv")
    te_meta = pd.read_csv(DATA_DIR / "test_metadata.csv")

    # Define the provided feature files
    feat_files = {
        "color": "color_histogram.csv",
        "hog": "hog_pca.csv",
        "additional": "additional_features.csv",
    }

    # Load the provided features into a dictionary
    feats = {
        name: pd.read_csv(DATA_DIR / filename)
        for name, filename in feat_files.items()
    }

    # Add extra handcrafted features if the file exists
    if EXTRA_PATH.exists():
        feats["extra"] = pd.read_csv(EXTRA_PATH)
        print(f"Extra features found: {feats['extra'].shape[1] - 1} dimensions")
    else:
        print("Extra features not found. Run extract_extra_features.py to include them.")

    # Add CNN features if the file exists
    if CNN_PATH.exists():
        feats["cnn"] = pd.read_csv(CNN_PATH)
        print(f"CNN features found: {feats['cnn'].shape[1] - 1} dimensions")
    else:
        print("CNN features not found. Run extract_cnn_features.py to include them.")

    return tr_meta, te_meta, feats


def build_matrices(meta, feats, feature_set):
    """Join selected feature groups by image_id and return one feature matrix."""

    # Store each selected feature matrix before combining them
    parts = []

    # Match each feature file to the metadata order using image_id
    for name in feature_set:
        df = feats[name].set_index("image_id").loc[meta["image_id"]]
        parts.append(df.values)

    # Combine all selected feature groups into one matrix
    return np.hstack(parts)


def get_class_names(tr_meta):
    """Return class names ordered by class_id."""

    # Extract class names in numeric class order
    return (
        tr_meta
        .drop_duplicates("class_id")
        .sort_values("class_id")["class_name"]
        .tolist()
    )


def make_feature_sets(feats):
    """Create feature-set combinations depending on which files are available."""

    # Start with the original features provided in the assignment
    feature_sets = {
        "provided": ["color", "hog", "additional"],
    }

    # Add handcrafted features if available
    if "extra" in feats:
        feature_sets["provided+extra"] = ["color", "hog", "additional", "extra"]

    # Add CNN-based feature combinations if available
    if "cnn" in feats:
        feature_sets["cnn_only"] = ["cnn"]
        feature_sets["provided+cnn"] = ["color", "hog", "additional", "cnn"]

        # Use all feature groups if both extra and CNN features exist
        if "extra" in feats:
            feature_sets["all"] = ["color", "hog", "additional", "extra", "cnn"]

    return feature_sets


def make_model_grids():
    """Return model pipelines and parameter grids for tuning."""

    # Each model is placed inside a pipeline so scaling is applied consistently
    return {
        "dummy_most_frequent": {
            "pipeline": Pipeline([
                ("scale", StandardScaler()),
                ("clf", DummyClassifier(strategy="most_frequent")),
            ]),
            "params": {},
        },

        "logreg": {
            "pipeline": Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                )),
            ]),
            "params": {
                "clf__C": [0.01, 0.1, 1, 10],
            },
        },

        "svm_rbf": {
            "pipeline": Pipeline([
                ("scale", StandardScaler()),
                ("clf", SVC(
                    kernel="rbf",
                    random_state=RANDOM_STATE,
                )),
            ]),
            "params": {
                "clf__C": [0.1, 1, 5, 10],
                "clf__gamma": ["scale", 0.001, 0.01, 0.1],
            },
        },

        "rf": {
            "pipeline": Pipeline([
                ("scale", StandardScaler()),
                ("clf", RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]),
            "params": {
                "clf__n_estimators": [200, 500],
                "clf__max_depth": [None, 20, 40],
                "clf__max_features": ["sqrt", "log2"],
            },
        },
    }


def tune_and_predict(X, y, pipeline, param_grid, cv):
    """Tune a model and return CV predictions from the best settings."""

    # Search over the parameter grid using cross-validation
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    # Fit the grid search on the training data
    search.fit(X, y)

    # Extract the best version of the model
    best_pipe = search.best_estimator_

    # Generate cross-validated predictions using the best model settings
    y_pred = cross_val_predict(
        best_pipe,
        X,
        y,
        cv=cv,
        n_jobs=-1,
    )

    return best_pipe, search.best_params_, y_pred


def plot_confusion(cm, labels, title, savepath):
    """Save a row-normalised confusion matrix."""

    # Convert raw counts into row-normalised proportions
    cm_n = cm / cm.sum(axis=1, keepdims=True)

    # Create the confusion matrix figure
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_n, vmin=0, vmax=1)

    # Add class labels to both axes
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Label the figure
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Write the numeric values inside each confusion matrix cell
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{cm_n[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    # Save the figure to disk
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()


def plot_accuracy_bars(results_df, savepath):
    """Save a bar chart comparing CV accuracy for all runs."""

    # Combine feature-set and model names for the x-axis labels
    labels = results_df["features"] + " + " + results_df["model"]
    values = results_df["accuracy"]

    # Create the accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(range(len(values)), values)

    # Label the chart
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Cross-validation accuracy")
    ax.set_title("Task 1 model and feature comparison")

    # Save the figure to disk
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()


def save_classification_report(y_true, y_pred, class_names, result_row, savepath):
    """Save a text classification report for one model."""

    # Generate precision, recall, F1-score, and support for each class
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=3,
    )

    # Save the model details and classification report
    with open(savepath, "w", encoding="utf-8") as f:
        f.write(f"Features: {result_row['features']}\n")
        f.write(f"Model: {result_row['model']}\n")
        f.write(f"Accuracy: {result_row['accuracy']:.4f}\n")
        f.write(f"Macro-F1: {result_row['macro_f1']:.4f}\n")
        f.write(f"Best parameters: {result_row['best_params']}\n\n")
        f.write(report)


def find_top_confusions(cm, class_names, top_n=3):
    """Return the largest off-diagonal confusion pairs."""

    # Store all cases where the true class and predicted class are different
    pairs = []

    # Loop through the confusion matrix and collect misclassification counts
    for true_id in range(cm.shape[0]):
        for pred_id in range(cm.shape[1]):
            if true_id != pred_id and cm[true_id, pred_id] > 0:
                pairs.append({
                    "true_id": true_id,
                    "pred_id": pred_id,
                    "true_name": class_names[true_id],
                    "pred_name": class_names[pred_id],
                    "count": cm[true_id, pred_id],
                })

    # Return the most common mistakes
    return sorted(pairs, key=lambda x: x["count"], reverse=True)[:top_n]


def save_misclassified_examples(
    tr_meta,
    y_true,
    y_pred,
    class_names,
    model_name,
    max_pairs=3,
    max_images_per_pair=6,
):
    """Save image grids for the most common misclassification pairs."""

    # Build a confusion matrix for this model
    cm = confusion_matrix(y_true, y_pred)

    # Find the most frequent class confusion pairs
    top_pairs = find_top_confusions(cm, class_names, top_n=max_pairs)

    # Create a folder for this model's misclassified examples
    model_dir = MISCLASS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save image examples for each major confusion pair
    for pair in top_pairs:
        mask = (y_true == pair["true_id"]) & (y_pred == pair["pred_id"])
        examples = tr_meta.loc[mask].head(max_images_per_pair)

        if examples.empty:
            continue

        # Create one row of example images
        n = len(examples)
        fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.5))

        if n == 1:
            axes = [axes]

        # Add each misclassified image to the grid
        for ax, row in zip(axes, examples.itertuples(index=False)):
            img_path = DATA_DIR / row.image_path
            img = Image.open(img_path).convert("RGB")

            ax.imshow(img)
            ax.axis("off")
            ax.set_title(str(row.image_id), fontsize=8)

        # Save the image grid to disk
        title = f"True {pair['true_name']} predicted {pair['pred_name']}"
        fig.suptitle(title)
        plt.tight_layout()

        filename = (
            f"true_{pair['true_name']}_pred_{pair['pred_name']}.png"
            .replace(" ", "_")
            .replace("/", "_")
        )

        plt.savefig(model_dir / filename, dpi=150)
        plt.close()


def save_best_model_outputs(
    tr_meta,
    y_train,
    class_names,
    results_df,
    all_predictions,
):
    """Save reports, confusion matrices, and error examples for each main model."""

    # Only save full analysis outputs for the three real models
    main_models = ["logreg", "svm_rbf", "rf"]

    # Find and save the best feature-set version of each model
    for model_name in main_models:
        model_rows = results_df[results_df["model"] == model_name]

        if model_rows.empty:
            continue

        # Select this model's best-performing feature set
        best_row = model_rows.sort_values("accuracy", ascending=False).iloc[0]
        key = (best_row["features"], best_row["model"])
        y_pred = all_predictions[key]

        # Create the confusion matrix
        cm = confusion_matrix(y_train, y_pred)

        # Save the confusion matrix figure
        plot_confusion(
            cm,
            class_names,
            title=f"{model_name}: {best_row['features']}",
            savepath=FIG_DIR / f"confusion_{model_name}.png",
        )

        # Save the classification report
        save_classification_report(
            y_train,
            y_pred,
            class_names,
            best_row,
            OUT_DIR / f"classification_report_{model_name}.txt",
        )

        # Save examples of the most common mistakes
        save_misclassified_examples(
            tr_meta,
            y_train,
            y_pred,
            class_names,
            model_name,
        )


def train_final_and_submit(
    tr_meta,
    te_meta,
    feats,
    feature_sets,
    best_row,
    model_grids,
):
    """Refit the best configuration on all training data and save test predictions."""

    # Identify the best model and feature set from cross-validation
    feature_name = best_row["features"]
    model_name = best_row["model"]

    # Build final train and test matrices using the best feature set
    X_train = build_matrices(tr_meta, feats, feature_sets[feature_name])
    X_test = build_matrices(te_meta, feats, feature_sets[feature_name])
    y_train = tr_meta["class_id"].values

    # Recreate the best model and apply the best hyperparameters
    final_pipe = model_grids[model_name]["pipeline"]
    final_pipe.set_params(**best_row["best_params"])

    # Train on the full labelled training set
    final_pipe.fit(X_train, y_train)

    # Predict labels for the unlabelled test set
    y_test_pred = final_pipe.predict(X_test)

    # Save predictions in Kaggle submission format
    submission = pd.DataFrame({
        "image_id": te_meta["image_id"].values,
        "class_id": y_test_pred,
    })

    submission_path = OUT_DIR / "task1_submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\nSaved final submission: {submission_path}")


def main():
    """Run the full Task 1 model comparison and final prediction pipeline."""

    # Create output folders if they do not already exist
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    MISCLASS_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata and feature files
    tr_meta, te_meta, feats = load_data()

    # Extract labels and class names
    y_train = tr_meta["class_id"].values
    class_names = get_class_names(tr_meta)

    # Create feature-set combinations and model grids
    feature_sets = make_feature_sets(feats)
    model_grids = make_model_grids()

    # Use stratified cross-validation so each fold keeps class balance
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Print basic dataset information
    print(f"\nClasses: {class_names}")
    print(f"Training rows: {len(tr_meta)}")
    print(f"Test rows: {len(te_meta)}\n")

    # Store summary results and predictions for later analysis
    results = []
    all_predictions = {}

    # Train every model on every feature set
    for feature_name, feature_group in feature_sets.items():
        X = build_matrices(tr_meta, feats, feature_group)
        print(f"Feature set: {feature_name} | shape={X.shape}")

        for model_name, spec in model_grids.items():
            start = time.time()

            # Tune model and generate cross-validated predictions
            best_pipe, best_params, y_pred = tune_and_predict(
                X,
                y_train,
                spec["pipeline"],
                spec["params"],
                cv,
            )

            # Calculate evaluation metrics
            acc = accuracy_score(y_train, y_pred)
            macro_f1 = f1_score(y_train, y_pred, average="macro")

            print(
                f"  {model_name:18s} "
                f"acc={acc:.4f} "
                f"macroF1={macro_f1:.4f} "
                f"time={time.time() - start:.1f}s"
            )

            # Save this run's summary results
            results.append({
                "features": feature_name,
                "model": model_name,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "best_params": best_params,
            })

            # Save predictions so confusion matrices can be made later
            all_predictions[(feature_name, model_name)] = y_pred

        print()

    # Convert results into a sorted dataframe
    results_df = (
        pd.DataFrame(results)
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )

    # Save all cross-validation results and the accuracy comparison chart
    results_df.to_csv(OUT_DIR / "cv_results.csv", index=False)
    plot_accuracy_bars(results_df, FIG_DIR / "accuracy_comparison.png")

    print("\nCross-validation results:")
    print(results_df.to_string(index=False))

    # Select the best overall model and feature set
    best_row = results_df.iloc[0]

    print("\nBest overall configuration:")
    print(best_row)

    # Get predictions for the best overall model
    y_pred_best = all_predictions[(best_row["features"], best_row["model"])]
    cm_best = confusion_matrix(y_train, y_pred_best)

    # Save confusion matrix and report for the best overall model
    plot_confusion(
        cm_best,
        class_names,
        title=f"Best: {best_row['features']} + {best_row['model']}",
        savepath=FIG_DIR / "confusion_matrix_best.png",
    )

    save_classification_report(
        y_train,
        y_pred_best,
        class_names,
        best_row,
        OUT_DIR / "classification_report_best.txt",
    )

    # Save reports, confusion matrices, and error examples for each main model
    save_best_model_outputs(
        tr_meta,
        y_train,
        class_names,
        results_df,
        all_predictions,
    )

    # Train the best model on all training data and create Kaggle test predictions
    train_final_and_submit(
        tr_meta,
        te_meta,
        feats,
        feature_sets,
        best_row,
        model_grids,
    )


# -----------------------------
# Run the script
# -----------------------------

if __name__ == "__main__":
    main()