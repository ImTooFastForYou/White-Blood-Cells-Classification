import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import loguniform
from dotenv import load_dotenv

from XGBoost2 import *


# ============================================================
#  TRAINING MODEL (SVM)
# ============================================================
def train_and_evaluate_svm(df_features, metadata_csv_path):
    """Fuse with labels, trains an SVM classifier and displays results."""

    # Merge features with labels
    df_labels = pd.read_csv(metadata_csv_path)
    df_final = pd.merge(df_features, df_labels, on="ID")

    # Build X and y
    feature_cols = [c for c in df_final.columns if c not in ("label", "ID", "image_id")]
    X = df_final[feature_cols]
    y = df_final["label"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=0, stratify=y_encoded
    )

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, random_state=0)),
        ]
    )

    # Hyperparameter search space
    param_distributions = {
        "svm__C": loguniform(1e-2, 1e3),
        "svm__gamma": loguniform(1e-4, 1e1),
    }

    print("\nSearching best SVM hyperparameters...")
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="f1_macro",
        cv=3,
        random_state=0,
        verbose=1,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train, svm__sample_weight=sample_weights)

    best_pipeline = random_search.best_estimator_

    print("\nBest parameters found:")
    print(random_search.best_params_)

    # Evaluation
    y_pred = best_pipeline.predict(X_test)
    print(f"\nF1-Score Macro: {f1_score(y_test, y_pred, average='macro'):.4f}")

    print("\n=== Classification Report ===")
    print(
        classification_report(
            encoder.inverse_transform(y_test),
            encoder.inverse_transform(y_pred),
            zero_division=0,
        )
    )

    svm_model = best_pipeline.named_steps["svm"]
    scaler = best_pipeline.named_steps["scaler"]

    if svm_model.kernel == "linear":
        importances = pd.Series(
            abs(svm_model.coef_[0]), index=feature_cols
        ).sort_values(ascending=True)
        title = "SVM Linear — |coefficient| per feature"
    else:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            best_pipeline,
            X_test,
            y_test,
            scoring="f1_macro",
            n_repeats=10,
            random_state=0,
            n_jobs=-1,
        )
        importances = pd.Series(
            result.importances_mean, index=feature_cols
        ).sort_values(ascending=True)
        title = "SVM RBF — Permutation importance per feature"

    plt.figure(figsize=(10, 6))
    importances.plot(kind="barh", color="steelblue")
    plt.title(title)
    plt.xlabel("Importance score")
    plt.tight_layout()
    plt.savefig("feature_importances_svm.png")
    plt.close()
    print("\nFeature importance plot saved to feature_importances_svm.png")

    return best_pipeline, encoder


# ============================================================
#  MAIN SCRIPT
# ============================================================
if __name__ == "__main__":
    load_dotenv()

    FOLDER_PATH = os.getenv("TRAIN_FOLDER_PATH")
    CSV_PATH = os.getenv("METADATA_CSV_PATH")

    if not FOLDER_PATH or not CSV_PATH:
        raise ValueError(
            "ERROR: TRAIN_FOLDER_PATH or METADATA_CSV_PATH not defined in .env !"
        )

    CROP_PARAMS = (104, 104, 159)
    THRESHOLDS = (0.68, 0.92, 0.08, 0.10, 0.97)

    # Comment once features are computed
    dataset_features = build_feature_dataset(FOLDER_PATH, CROP_PARAMS, THRESHOLDS)
    dataset_features.to_csv("features_SVM2.csv", index=False)

    dataset_features = pd.read_csv("features_XGBoost2.csv")

    dataset_features = dataset_features.loc[
        :, ~dataset_features.columns.str.contains("^Unnamed")
    ]
    dataset_features = dataset_features.dropna(axis=1, how="all")

    trained_pipeline, label_encoder = train_and_evaluate_svm(dataset_features, CSV_PATH)
