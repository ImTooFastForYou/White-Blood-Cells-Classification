import os
import glob
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from skimage import color, filters
from skimage.io import imread
from skimage.measure import regionprops_table, label
from skimage.morphology import remove_small_objects, remove_small_holes
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from skimage.feature import local_binary_pattern
from dotenv import load_dotenv


# ============================================================
#  EXTRACTION
# ============================================================
def extract_features_from_image(image_path, crop_params, thresholds):
    """Loads an image, segments it and returns a dataframe with features on one line."""

    # LOADING AND CROP
    img_rgb_full = imread(image_path)
    if img_rgb_full.ndim == 3 and img_rgb_full.shape[2] == 4:
        img_rgb_full = img_rgb_full[:, :, :3]

    cx, cy, csize = crop_params
    img_rgb = img_rgb_full[cy : cy + csize, cx : cx + csize].copy()
    img_float = (
        img_rgb.astype(np.float32) / 255.0
        if img_rgb.dtype == np.uint8
        else img_rgb.astype(np.float32)
    )

    # COLORS SPACE AND MASK
    img_hsv = color.rgb2hsv(img_float)
    H, S, V = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]

    h_min, h_max, s_min, v_min, v_max = thresholds
    purple_mask = (
        (H >= h_min) & (H <= h_max) & (S >= s_min) & (V >= v_min) & (V <= v_max)
    )

    purple_pixels = V[purple_mask]
    if len(purple_pixels) == 0:
        return pd.DataFrame()

    otsu_thresh = filters.threshold_otsu(purple_pixels)
    nuc_mask = purple_mask & (V < otsu_thresh)
    cyto_mask = purple_mask & (V >= otsu_thresh)

    # CLEANING
    nuc_clean = remove_small_holes(
        remove_small_objects(nuc_mask, max_size=50), max_size=50
    )
    cyto_clean = remove_small_holes(
        remove_small_objects(cyto_mask, max_size=50), max_size=50
    )

    # FEATURES
    _, nb_lobes = label(nuc_clean, return_num=True)

    V_int = (V * 255).astype(np.uint8)
    lbp = local_binary_pattern(V_int, P=8, R=1, method="uniform")
    texture_mean = np.mean(lbp[cyto_clean]) if np.any(cyto_clean) else 0.0

    nuc_labeled = nuc_clean.astype(int)
    cyto_labeled = cyto_clean.astype(int)

    props = ("area", "perimeter", "eccentricity", "solidity", "extent")

    nuc_feat = pd.DataFrame(
        regionprops_table(
            nuc_labeled, intensity_image=V, properties=props + ("mean_intensity",)
        )
    ).add_prefix("nuc_")
    cyto_feat = pd.DataFrame(
        regionprops_table(
            cyto_labeled, intensity_image=V, properties=props + ("mean_intensity",)
        )
    ).add_prefix("cyto_")

    # FINAL DATAFRAME
    df_cell = pd.concat([nuc_feat, cyto_feat], axis=1)

    if not df_cell.empty:
        df_cell["ratio_NC"] = df_cell["nuc_area"] / (df_cell["cyto_area"] + 1e-5)
        df_cell["ratio_intensity"] = df_cell["nuc_mean_intensity"] / (
            df_cell["cyto_mean_intensity"] + 1e-5
        )

        df_cell["nb_lobes"] = nb_lobes
        df_cell["cyto_texture_mean"] = texture_mean

        df_cell["nuc_circularity"] = (4 * np.pi * df_cell["nuc_area"]) / (
            df_cell["nuc_perimeter"] ** 2 + 1e-5
        )

        df_cell["ID"] = os.path.basename(image_path)

    return df_cell


# ============================================================
#  CREATE DATASET
# ============================================================
def build_feature_dataset(image_folder, crop_params, thresholds):
    """Loops on every images and creates a big DataFrame of features."""
    images = glob.glob(os.path.join(image_folder, "*.png"))
    every_cells = []

    print(f"Starting extraction on {len(images)} images...")
    for idx, image_path in enumerate(images):
        if idx % 100 == 0 and idx > 0:
            print(f"Progression : {idx} / {len(images)}")

        df_cell = extract_features_from_image(image_path, crop_params, thresholds)
        if not df_cell.empty:
            every_cells.append(df_cell)

    df_final = pd.concat(every_cells, ignore_index=True)
    print("Extraction done !")
    return df_final


# ============================================================
# TRAINING MODEL
# ============================================================
def train_and_evaluate_model(df_features, metadata_csv_path):
    """Fuse with Kaggle labels, trains XGBoost and displays results."""
    # Fusion
    df_labels = pd.read_csv(metadata_csv_path)
    df_final = pd.merge(df_features, df_labels, on="ID")

    # Creating X and y
    X = df_final.drop(["label", "ID"], axis=1)
    y = df_final["label"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=0, stratify=y_encoded
    )
    weight_samples = compute_sample_weight(class_weight="balanced", y=y_train)
    # Training
    print("\nTraining XGBoost Random Forest model...")
    param_distributions = {
        "max_depth": [4, 6, 8, 10, 12],
        "min_child_weight": [
            1,
            3,
            5,
            7,
        ],
        "subsample": [0.5, 0.7, 0.9],
        "colsample_bynode": [0.5, 0.7, 0.9],
    }

    # Base model initialisation
    rf_base = xgb.XGBRFClassifier(n_estimators=100, random_state=0, n_jobs=-1)

    # Starts random search
    print("\nSearching best hyperparameters...")
    random_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="f1_macro",
        cv=3,
        random_state=0,
        verbose=1,
    )

    random_search.fit(X_train, y_train, sample_weight=weight_samples)

    # Get best estimator
    rf_best = random_search.best_estimator_

    print("\nBest paramaters found :")
    print(random_search.best_params_)

    # Final evaluation
    y_pred = rf_best.predict(X_test)
    print(f"\nNew F1-Score Macro : {f1_score(y_test, y_pred, average='macro'):.4f}")

    print("\n=== Classification Report ===")
    print(
        classification_report(
            encoder.inverse_transform(y_test),
            encoder.inverse_transform(y_pred),
            zero_division=0,
        )
    )

    # Visualisation
    importances = pd.Series(
        rf_best.feature_importances_, index=X_train.columns
    ).sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    importances.plot(kind="barh", color="purple")
    plt.title("Caracteristics")
    plt.xlabel("Score")
    plt.tight_layout()
    plt.savefig("feature_importances_XGBoost2.png")
    plt.close()

    return rf_best, encoder


# ============================================================
#  MAIN SCRIPT
# ============================================================
if __name__ == "__main__":
    load_dotenv()
    # Parameters
    FOLDER_PATH = os.getenv("TRAIN_FOLDER_PATH")
    CSV_PATH = os.getenv("METADATA_CSV_PATH")
    if not FOLDER_PATH or not CSV_PATH:
        raise ValueError(
            "ERROR: TRAIN_IMAGE_FOLDER or METADATA_CSV_PATH not defined in .env !"
        )
    CROP_PARAMS = (104, 104, 159)
    THRESHOLDS = (0.68, 0.92, 0.08, 0.10, 0.97)

    # Dataset
    # Compute once features are computed
    dataset_features = build_feature_dataset(FOLDER_PATH, CROP_PARAMS, THRESHOLDS)
    dataset_features.to_csv("features_XGBoost2.csv", index=False)
    dataset_features = pd.read_csv("features_XGBoost2.csv")

    # Training
    modele_entraine, label_encoder = train_and_evaluate_model(
        dataset_features, CSV_PATH
    )
