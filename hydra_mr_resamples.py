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
    max_memory_mb = 50000  # 50GB available
    memory_safety_threshold = 40000  # Stop at 40GB (80% usage for breathing room)
    memory_check_interval = 5  # Check memory every 5 tasks
    skip_large_mb = 10000  # Increased threshold
    skip_large_features = 50000  # Increased threshold
    output_file = "ucr_baseline_results_wide.csv"
    model_dir = "baseline_models_resamples"
    
    # Performance optimizations
    use_parallel = True  # Enable parallel processing
    adaptive_workers = True  # Dynamically adjust workers based on memory
    aggressive_params = True  # Use maximum transformer parameters


def get_available_memory_mb():
    """Get available system memory in MB."""
    return psutil.virtual_memory().available / (1024 * 1024)

def cleanup():
    """Aggressive garbage collection."""
    gc.collect()
    gc.collect()


def memory_usage_mb():
    """Return current memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def check_memory_safe():
    """Check if memory usage is safe to continue processing."""
    current_usage = memory_usage_mb()
    available = get_available_memory_mb()
    
    # Return True if we're below safety threshold and have enough available
    return current_usage < Config.memory_safety_threshold and available > 5000


def adaptive_cleanup(force=False):
    """
    Adaptive garbage collection based on memory pressure.
    
    Args:
        force: Force aggressive cleanup regardless of memory status
    """
    current_usage = Config.memory_usage_mb()
    available = get_available_memory_mb()
    
    # Aggressive cleanup if memory is getting tight
    if force or current_usage > Config.memory_safety_threshold * 0.9 or available < 8000:
        gc.collect()
        gc.collect()
        gc.collect()  # Triple collection for safety
        import time
        time.sleep(0.5)  # Give system time to breathe
    elif current_usage > Config.memory_safety_threshold * 0.7:
        gc.collect()
        gc.collect()
    else:
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
    Memory-aware parameter selection to prevent crashes.
    """
    available_mem = get_available_memory_mb()
    
    # Adjust parameters based on both dataset size AND available memory
    # If memory is tight, scale back even for small datasets
    memory_factor = min(1.0, available_mem / 20000)  # Scale down if < 20GB available
    
    if size_mb > 5000 or available_mem < 15000:
        # Very large datasets or low memory - conservative
        multirocket = MultiRocket(n_kernels=int(2000 * memory_factor), n_jobs=-1, random_state=42)
        hydra = HydraTransformer(n_kernels=max(2, int(4 * memory_factor)), n_groups=32, random_state=42)
    elif size_mb > 1000 or available_mem < 25000:
        # Large datasets or medium memory
        multirocket = MultiRocket(n_kernels=int(5000 * memory_factor), n_jobs=-1, random_state=42)
        hydra = HydraTransformer(n_kernels=max(4, int(8 * memory_factor)), n_groups=64, random_state=42)
    elif size_mb > 500 or available_mem < 35000:
        # Medium datasets with good memory
        multirocket = MultiRocket(n_kernels=int(10000 * memory_factor), n_jobs=-1, random_state=42)
        hydra = HydraTransformer(n_kernels=max(8, int(16 * memory_factor)), n_groups=128, random_state=42)
    else:
        # Small datasets with plenty of memory - maximum parameters
        multirocket = MultiRocket(n_kernels=int(20000 * memory_factor), n_jobs=-1, random_state=42)
        hydra = HydraTransformer(n_kernels=max(16, int(32 * memory_factor)), n_groups=256, random_state=42)

    scaler_std, scaler_hydra = StandardScaler(), StandardScaler()
    Xt_hydra = scaler_hydra.fit_transform(hydra.fit_transform(X_sample))
    Xt_multi = scaler_std.fit_transform(multirocket.fit_transform(X_sample))
    features = np.concatenate([Xt_hydra, Xt_multi], axis=1)

    return multirocket, hydra, scaler_std, scaler_hydra, features


def process_dataset_resample(args):
    """
    Process a single dataset-resample combination (for parallel execution).
    Memory-safe with cleanup after each task.
    
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
        # Check memory before starting
        if not check_memory_safe():
            print(f"⚠️  Memory pressure detected, waiting for {dataset} resample {resample_id}...")
            import time
            time.sleep(5)
            adaptive_cleanup(force=True)
            
            if not check_memory_safe():
                print(f"❌ Insufficient memory for {dataset} resample {resample_id}, skipping...")
                return (dataset, resample_id, np.nan)
        
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

        # Clear dataframes from memory
        del train_df, test_df, full_df
        adaptive_cleanup()

        # Fit transformers and classifier on a sample
        train_size, _, n_features = dataset_info(folder, dataset)
        multirocket, hydra, scaler_std, scaler_hydra, X_fit = fit_transformers(
            X_train[:min(100, len(X_train))], n_features, train_size
        )
        clf = RidgeClassifier().fit(X_fit, y_train[:min(100, len(X_train))])

        # Transform and train on full training data
        Xt_hydra = scaler_hydra.transform(hydra.transform(X_train))
        Xt_multi = scaler_std.transform(multirocket.transform(X_train))
        X_train_features = np.concatenate([Xt_hydra, Xt_multi], axis=1)
        
        # Clear intermediate arrays
        del Xt_hydra, Xt_multi, X_train
        adaptive_cleanup()
        
        clf.fit(X_train_features, y_train)
        del X_train_features
        adaptive_cleanup()

        # Transform and evaluate on test data
        Xt_hydra = scaler_hydra.transform(hydra.transform(X_test))
        Xt_multi = scaler_std.transform(multirocket.transform(X_test))
        X_test_features = np.concatenate([Xt_hydra, Xt_multi], axis=1)
        
        # Clear intermediate arrays
        del Xt_hydra, Xt_multi, X_test
        adaptive_cleanup()
        
        acc = accuracy_score(y_test, clf.predict(X_test_features))

        # Save model
        joblib.dump({
            "clf": clf,
            "hydra": hydra,
            "multirocket": multirocket,
            "scaler_std": scaler_std,
            "scaler_hydra": scaler_hydra
        }, model_file)

        # Aggressive cleanup before returning
        del X_test_features, clf, hydra, multirocket, scaler_std, scaler_hydra
        adaptive_cleanup(force=True)

        return (dataset, resample_id, acc)
    
    except Exception as e:
        print(f"❌ Error processing {dataset} resample {resample_id}: {str(e)}")
        adaptive_cleanup(force=True)
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
    Memory-aware execution with dynamic worker adjustment.
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
    print(f"💾 Max memory: {Config.max_memory_mb}MB | Safety threshold: {Config.memory_safety_threshold}MB")
    print(f"🔧 Adaptive workers: {Config.adaptive_workers}")

    # Create all tasks (dataset, resample combinations)
    tasks = []
    for dataset in datasets:
        for r in range(1, n_resamples + 1):
            resample_model_dir = os.path.join(model_dir, f"resample_{r}")
            tasks.append((folder, dataset, r, resample_model_dir))

    print(f"📋 Total tasks to process: {len(tasks)}")

    # Process tasks in parallel with memory monitoring
    completed_tasks = 0
    current_dataset = None
    dataset_results = {}
    memory_warnings = 0
    
    # Adjust workers dynamically based on available memory
    current_workers = max_workers
    if Config.adaptive_workers:
        available_mem = get_available_memory_mb()
        if available_mem < 20000:
            current_workers = max(3, max_workers // 2)
            print(f"⚠️  Low memory detected, reducing workers to {current_workers}")

    with ProcessPoolExecutor(max_workers=current_workers) as executor:
        # Submit tasks in batches to control memory
        batch_size = max_workers * 3  # 3x workers for good parallelism
        task_batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
        
        for batch_idx, batch in enumerate(task_batches):
            print(f"\n📦 Processing batch {batch_idx + 1}/{len(task_batches)} ({len(batch)} tasks)")
            
            # Check memory before starting batch
            current_mem = memory_usage_mb()
            available_mem = get_available_memory_mb()
            print(f"   💾 Memory: {current_mem:.0f}MB used | {available_mem:.0f}MB available")
            
            if not check_memory_safe():
                print(f"   ⚠️  Memory pressure detected, cleaning up...")
                adaptive_cleanup(force=True)
                import time
                time.sleep(3)
                
                # Reduce workers if memory is consistently tight
                if Config.adaptive_workers and memory_warnings > 2:
                    current_workers = max(2, current_workers - 1)
                    print(f"   🔧 Reducing workers to {current_workers}")
                    executor._max_workers = current_workers
                
                memory_warnings += 1
            else:
                memory_warnings = max(0, memory_warnings - 1)
            
            # Submit batch tasks
            future_to_task = {executor.submit(process_dataset_resample, task): task for task in batch}
            
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
                
                # Memory check and cleanup every N tasks
                if completed_tasks % Config.memory_check_interval == 0:
                    current_mem = memory_usage_mb()
                    if current_mem > Config.memory_safety_threshold * 0.8:
                        adaptive_cleanup(force=True)
                    
                    # Save periodically
                    df.to_csv(output_file, index=False)
            
            # Cleanup between batches
            adaptive_cleanup(force=True)
            print(f"   ✅ Batch {batch_idx + 1} complete")
    
    # Final save and cleanup
    df.to_csv(output_file, index=False)
    adaptive_cleanup(force=True)
    
    # Calculate final statistics
    final_mean = df["MeanAccuracy"].dropna().mean() if len(df) else None
    print("\n" + "="*60)
    print("✅ All datasets complete!")
    if final_mean is not None:
        print(f"🏆 Overall Mean Accuracy: {final_mean:.4f}")
        print(f"💾 Peak memory warnings: {memory_warnings}")
    else:
        print("⚠️ No valid results")
    print("="*60)
    
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