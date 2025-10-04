import os
import gc
import psutil
import numpy as np
import pandas as pd
import joblib
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from aeon.transformations.collection.convolution_based import MultiRocket, HydraTransformer

warnings.filterwarnings("ignore")


class Config:
    """Central configuration for evaluation - Optimized for 50GB RAM, 9 threads"""
    n_resamples = 30
    max_workers = 9  # Full thread utilization
    max_memory_mb = 40000  # 50GB available
    skip_large_mb = 10000  # Increased threshold
    skip_large_features = 50000  # Increased threshold
    output_file = "ucr_baseline_results_wide.csv"
    model_dir = "baseline_models_resamples"
    
    # Performance optimizations
    use_parallel = True  # Enable parallel processing
    prefetch_datasets = True  # Load datasets in advance
    aggressive_params = True  # Use maximum transformer parameters


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
    """
    Fit Hydra and MultiRocket with scaling, optimized for high-resource machine.
    Uses aggressive parameters to maximize quality and speed.
    """
    # With 50GB RAM, we can use much more aggressive parameters
    if size_mb > 5000:
        # Very large datasets - conservative but still aggressive
        multirocket = MultiRocket(n_kernels=2000, n_jobs=-1, random_state=42)
        hydra = HydraTransformer(n_kernels=4, n_groups=32, random_state=42)
    elif size_mb > 1000:
        # Large datasets - more aggressive
        multirocket = MultiRocket(n_kernels=5000, n_jobs=-1, random_state=42)
        hydra = HydraTransformer(n_kernels=8, n_groups=64, random_state=42)
    elif size_mb > 500:
        # Medium datasets - very aggressive
        multirocket = MultiRocket(n_kernels=10000, n_jobs=-1, random_state=42)
        hydra = HydraTransformer(n_kernels=16, n_groups=128, random_state=42)
    else:
        # Small datasets - maximum parameters
        multirocket = MultiRocket(n_kernels=20000, n_jobs=-1, random_state=42)
        hydra = HydraTransformer(n_kernels=32, n_groups=256, random_state=42)

    scaler_std, scaler_hydra = StandardScaler(), StandardScaler()
    Xt_hydra = scaler_hydra.fit_transform(hydra.fit_transform(X_sample))
    Xt_multi = scaler_std.fit_transform(multirocket.fit_transform(X_sample))
    features = np.concatenate([Xt_hydra, Xt_multi], axis=1)

    return multirocket, hydra, scaler_std, scaler_hydra, features


def process_dataset_resample(args):
    """
    Process a single dataset-resample combination (for parallel execution).
    
    Args:
        args: Tuple of (folder, dataset, resample_id, model_dir)
    
    Returns:
        Tuple of (dataset, resample_id, accuracy)
    """
    folder, dataset, resample_id, model_dir = args
    
    # Set random seed for reproducibility
    np.random.seed(resample_id)

    # Setup model directory and check if model exists
    dataset_model_dir = os.path.join(model_dir, dataset)
    model_file = os.path.join(dataset_model_dir, "clf.pkl")
    os.makedirs(dataset_model_dir, exist_ok=True)
    
    if os.path.exists(model_file):
        return (dataset, resample_id, None)

    try:
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
            X_train[:min(100, len(X_train))], n_features, train_size
        )
        clf = RidgeClassifier().fit(X_fit, y_train[:min(100, len(X_train))])

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

        return (dataset, resample_id, acc)
    
    except Exception as e:
        print(f"❌ Error processing {dataset} resample {resample_id}: {str(e)}")
        return (dataset, resample_id, np.nan)


def process_dataset(folder, dataset, resample_id, model_dir):
    """
    Single-threaded version (for compatibility).
    Wrapper around process_dataset_resample for non-parallel execution.
    """
    _, _, acc = process_dataset_resample((folder, dataset, resample_id, model_dir))
    return acc


def evaluate_all_parallel(folder, output_file=Config.output_file, model_dir=Config.model_dir, 
                          n_resamples=Config.n_resamples, max_workers=Config.max_workers):
    """
    Parallel evaluation of all datasets with resampling.
    Utilizes all available CPU cores for maximum speed.
    """
    # Load or create results table
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        processed = set(df["Dataset"].tolist())
        print(f"📂 Resuming from existing results: {len(processed)} datasets already done")
    else:
        cols = ["Dataset"] + [f"Resample_{i}" for i in range(1, n_resamples + 1)] + ["MeanAccuracy"]
        df, processed = pd.DataFrame(columns=cols), set()
        print(f"📂 Starting fresh run, output will be saved to {output_file}")

    # Get all datasets
    datasets = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    datasets = [d for d in datasets if d not in processed]
    
    if not datasets:
        print("✅ All datasets already processed!")
        return df, df["MeanAccuracy"].dropna().mean() if len(df) else None

    # Sort datasets by size (smallest first for better scheduling)
    datasets.sort(key=lambda d: sum(dataset_info(folder, d)[:2]))

    print(f"🚀 Processing {len(datasets)} datasets with {max_workers} parallel workers")
    print(f"💾 Available memory: {Config.max_memory_mb}MB")
    print(f"🔧 Aggressive parameters: {Config.aggressive_params}")

    # Create all tasks (dataset, resample combinations)
    tasks = []
    for dataset in datasets:
        for r in range(1, n_resamples + 1):
            resample_model_dir = os.path.join(model_dir, f"resample_{r}")
            tasks.append((folder, dataset, r, resample_model_dir))

    print(f"📋 Total tasks to process: {len(tasks)}")

    # Process tasks in parallel
    completed_tasks = 0
    current_dataset = None
    dataset_results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(process_dataset_resample, task): task for task in tasks}
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            dataset, resample_id, acc = future.result()
            completed_tasks += 1
            
            # Initialize dataset results if needed
            if dataset not in dataset_results:
                dataset_results[dataset] = {}
            
            dataset_results[dataset][resample_id] = acc
            
            # Progress update
            if dataset != current_dataset:
                current_dataset = dataset
                print(f"\n🔄 Processing: {dataset}")
            
            status = f"✅ {acc:.4f}" if acc is not None else "⚠️  Exists"
            print(f"   Resample {resample_id}/{n_resamples}: {status} "
                  f"[{completed_tasks}/{len(tasks)} total]")
            
            # Update and save results after each task
            if dataset in df["Dataset"].values:
                df.loc[df["Dataset"] == dataset, f"Resample_{resample_id}"] = acc
                resample_cols = [f"Resample_{i}" for i in range(1, n_resamples + 1)]
                df.loc[df["Dataset"] == dataset, "MeanAccuracy"] = \
                    df.loc[df["Dataset"] == dataset, resample_cols].mean(axis=1, skipna=True).values[0]
            else:
                row = {"Dataset": dataset, **{f"Resample_{i}": np.nan for i in range(1, n_resamples + 1)},
                       "MeanAccuracy": np.nan}
                row[f"Resample_{resample_id}"] = acc
                row["MeanAccuracy"] = acc if acc is not None else np.nan
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            
            # Save every 10 tasks
            if completed_tasks % 10 == 0:
                df.to_csv(output_file, index=False)
    
    # Final save
    df.to_csv(output_file, index=False)
    
    # Calculate final statistics
    final_mean = df["MeanAccuracy"].dropna().mean() if len(df) else None
    print("\n" + "="*60)
    print("✅ All datasets complete!")
    if final_mean is not None:
        print(f"🏆 Overall Mean Accuracy: {final_mean:.4f}")
    else:
        print("⚠️ No valid results")
    print("="*60)
    
    cleanup()
    return df, final_mean


def evaluate_all(folder, output_file=Config.output_file, model_dir=Config.model_dir, 
                 n_resamples=Config.n_resamples):
    """
    Main evaluation function - automatically uses parallel processing if enabled.
    """
    if Config.use_parallel:
        return evaluate_all_parallel(folder, output_file, model_dir, n_resamples, Config.max_workers)
    else:
        # Original sequential implementation (kept for compatibility)
        return evaluate_all_sequential(folder, output_file, model_dir, n_resamples)


def evaluate_all_sequential(folder, output_file, model_dir, n_resamples):
    """Original sequential evaluation (for compatibility)."""
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        processed = set(df["Dataset"].tolist())
        print(f"📂 Resuming from existing results: {len(processed)} datasets already done")
    else:
        cols = ["Dataset"] + [f"Resample_{i}" for i in range(1, n_resamples + 1)] + ["MeanAccuracy"]
        df, processed = pd.DataFrame(columns=cols), set()
        print(f"📂 Starting fresh run, output will be saved to {output_file}")

    datasets = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    datasets.sort(key=lambda d: sum(dataset_info(folder, d)[:2]))

    print(f"🚀 Processing {len(datasets)} datasets sequentially")

    for dataset in datasets:
        if dataset in processed:
            print(f"⏭️  Skipping {dataset} (already processed)")
            continue

        print(f"\n🔄 Dataset: {dataset}")
        res_accs = []

        for r in range(1, n_resamples + 1):
            print(f"   ▶️ Resample {r}/{n_resamples}")
            acc = process_dataset(folder, dataset, r, os.path.join(model_dir, f"resample_{r}"))

            if acc is not None:
                print(f"      ✅ Accuracy: {acc:.4f}")
            else:
                print(f"      ⚠️ Skipped (model already exists)")

            res_accs.append(acc if acc is not None else np.nan)

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