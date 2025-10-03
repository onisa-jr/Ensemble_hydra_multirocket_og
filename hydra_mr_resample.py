import os
import gc
import psutil
import numpy as np
import pandas as pd
import joblib
import warnings
from tqdm.notebook import tqdm
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from aeon.transformations.collection.convolution_based import MultiRocket, HydraTransformer

warnings.filterwarnings("ignore")


class Config:
    """Central configuration for evaluation"""
    n_resamples = 30
    max_memory_mb = 40000
    skip_large_mb = 2000
    skip_large_features = 10000
    output_file = "ucr_baseline_results_wide.csv"
    model_dir = "baseline_models_resamples"


def memory_usage_mb():
    """Return current memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def cleanup():
    """Aggressive garbage collection."""
    gc.collect()
    gc.collect()


def dataset_info(folder, name):
    """Return (train_size_mb, test_size_mb, n_features)."""
    train_file = os.path.join(folder, name, f"{name}_TRAIN.tsv")
    test_file = os.path.join(folder, name, f"{name}_TEST.tsv")

    train_size = os.path.getsize(train_file) / (1024 * 1024) if os.path.exists(train_file) else 0
    test_size = os.path.getsize(test_file) / (1024 * 1024) if os.path.exists(test_file) else 0

    if train_size > 0:
        sample = pd.read_csv(train_file, sep="\t", header=None, nrows=1)
        n_features = sample.shape[1] - 1
    else:
        n_features = 0

    return train_size, test_size, n_features


def fit_transformers(X_sample, n_features, size_mb):
    """Fit Hydra and MultiRocket with scaling, using parameters adapted to dataset size."""
    if size_mb > 1000:
        multirocket = MultiRocket(n_kernels=1000, n_jobs=1, random_state=42)
        hydra = HydraTransformer(n_kernels=2, n_groups=8, random_state=42)
    elif size_mb > 500:
        multirocket = MultiRocket(n_kernels=2000, n_jobs=2, random_state=42)
        hydra = HydraTransformer(n_kernels=2, n_groups=16, random_state=42)
    elif size_mb > 100:
        multirocket = MultiRocket(n_kernels=5000, n_jobs=-1, random_state=42)
        hydra = HydraTransformer(n_kernels=4, n_groups=32, random_state=42)
    else:
        multirocket = MultiRocket(n_jobs=2, random_state=42)
        hydra = HydraTransformer(n_kernels=8, n_groups=64, random_state=42)

    scaler_std, scaler_hydra = StandardScaler(), StandardScaler()
    Xt_hydra = scaler_hydra.fit_transform(hydra.fit_transform(X_sample))
    Xt_multi = scaler_std.fit_transform(multirocket.fit_transform(X_sample))
    features = np.concatenate([Xt_hydra, Xt_multi], axis=1)

    return multirocket, hydra, scaler_std, scaler_hydra, features


def process_dataset(folder, dataset, resample_id, model_dir):
    # Set random seed for reproducibility
    np.random.seed(resample_id)

    # Setup model directory and check if model exists
    dataset_model_dir = os.path.join(model_dir, dataset)
    model_file = os.path.join(dataset_model_dir, "clf.pkl")
    os.makedirs(dataset_model_dir, exist_ok=True)
    if os.path.exists(model_file):
        return None

    # Load train and test data
    train_file = os.path.join(folder, dataset, f"{dataset}_TRAIN.tsv")
    test_file = os.path.join(folder, dataset, f"{dataset}_TEST.tsv")
    train_df = pd.read_csv(train_file, sep="\t", header=None)
    test_df = pd.read_csv(test_file, sep="\t", header=None)

    # Combine and shuffle data
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=resample_id).reset_index(drop=True)

    # Split back into train and test sets
    n_train, n_test = len(train_df), len(test_df)
    train_df = full_df.iloc[:n_train].reset_index(drop=True)
    test_df = full_df.iloc[n_train:n_train + n_test].reset_index(drop=True)

    # Prepare feature and label arrays
    y_train = train_df.iloc[:, 0].values
    X_train = train_df.iloc[:, 1:].values.reshape(train_df.shape[0], 1, train_df.shape[1] - 1)
    y_test = test_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values.reshape(test_df.shape[0], 1, test_df.shape[1] - 1)

    # Fit transformers and classifier on a sample
    train_size, _, n_features = dataset_info(folder, dataset)
    multirocket, hydra, scaler_std, scaler_hydra, X_fit = fit_transformers(
        X_train[:50], n_features, train_size
    )
    clf = RidgeClassifier().fit(X_fit, y_train[:50])

    # Transform and train on full training data
    Xt_hydra = scaler_hydra.transform(hydra.transform(X_train))
    Xt_multi = scaler_std.transform(multirocket.transform(X_train))
    clf.fit(np.concatenate([Xt_hydra, Xt_multi], axis=1), y_train)

    # Transform and evaluate on test data
    Xt_hydra = scaler_hydra.transform(hydra.transform(X_test))
    Xt_multi = scaler_std.transform(multirocket.transform(X_test))
    X_test_features = np.concatenate([Xt_hydra, Xt_multi], axis=1)
    acc = accuracy_score(y_test, clf.predict(X_test_features))

    # Save model
    joblib.dump({
        "clf": clf,
        "hydra": hydra,
        "multirocket": multirocket,
        "scaler_std": scaler_std,
        "scaler_hydra": scaler_hydra
    }, model_file)

    # Cleanup
    cleanup()

    return acc


def evaluate_all(folder, output_file=Config.output_file, model_dir=Config.model_dir, n_resamples=Config.n_resamples):
    """Evaluate all datasets with resampling (rows: dataset, columns: resamples)."""

    # Load or create results table
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        processed = set(df["Dataset"].tolist())
        print(f"📂 Resuming from existing results: {len(processed)} datasets already done")
    else:
        cols = ["Dataset"] + [f"Resample_{i}" for i in range(1, n_resamples + 1)] + ["MeanAccuracy"]
        df, processed = pd.DataFrame(columns=cols), set()
        print(f"📂 Starting fresh run, output will be saved to {output_file}")

    # Sort datasets by size (smallest first)
    datasets = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    datasets.sort(key=lambda d: sum(dataset_info(folder, d)[:2]))

    print(f"🚀 Processing {len(datasets)} datasets in total")

    for dataset in datasets:
        if dataset in processed:
            print(f"⏭️  Skipping {dataset} (already processed)")
            continue

        print(f"\n🔄 Dataset: {dataset}")
        res_accs = []

        for r in range(1, n_resamples + 1):
            print(f"   ▶️ Resample {r}/{n_resamples} (seed={r})")
            acc = process_dataset(folder, dataset, r, os.path.join(model_dir, f"resample_{r}"))

            if acc is not None:
                print(f"      ✅ Accuracy: {acc:.4f}")
            else:
                print(f"      ⚠️ Skipped (model already exists)")

            res_accs.append(acc if acc is not None else np.nan)

            # Save partial results live (after each resample)
            if dataset in df["Dataset"].values:
                df.loc[df["Dataset"] == dataset, f"Resample_{r}"] = res_accs[-1]
                df.loc[df["Dataset"] == dataset, "MeanAccuracy"] = np.nanmean(res_accs)
            else:
                row = {"Dataset": dataset, **{f"Resample_{i + 1}": np.nan for i in range(n_resamples)},
                       "MeanAccuracy": np.nan}
                row[f"Resample_{r}"] = res_accs[-1]
                row["MeanAccuracy"] = np.nanmean(res_accs)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

            df.to_csv(output_file, index=False)

        print(f"   📊 Finished {dataset} → Mean accuracy: {np.nanmean(res_accs):.4f}")
        cleanup()

    final_mean = df["MeanAccuracy"].dropna().mean() if len(df) else None
    print("\n✅ All datasets complete")
    if final_mean is not None:
        print(f"🏆 Overall Mean Accuracy: {final_mean:.4f}")
    else:
        print("⚠️ No valid results")

    return df, final_mean


if __name__ == "__main__":
    pass
