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
    """Memory-optimized configuration for 45GB RAM, 9 threads"""
    n_resamples = 30
    max_workers = 6  # Reduced from 9 to leave memory headroom
    max_memory_mb = 35000  # Leave 10GB for system
    skip_large_mb = 5000   # More conservative threshold
    skip_large_features = 20000  # Reduced threshold
    output_file = "ucr_baseline_results_wide.csv"
    model_dir = "baseline_models_resamples"
    
    # Memory optimizations
    use_parallel = True
    batch_size = 3  # Process datasets in smaller batches
    chunk_processing = True  # Enable chunking for large datasets
    memory_monitor = True  # Monitor memory usage


def memory_usage_mb():
    """Return current memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def memory_safe_cleanup():
    """Memory-aware garbage collection."""
    gc.collect()
    if memory_usage_mb() > Config.max_memory_mb * 0.8:  # If using >80% memory
        print(f"⚠️  High memory usage: {memory_usage_mb():.0f}MB, forcing cleanup")
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


def fit_transformers_memory_safe(X_sample, n_features, size_mb):
    """
    Fit Hydra and MultiRocket with memory-conscious parameters.
    """
    # More conservative parameters to save memory
    if size_mb > 2000:
        # Very large datasets - very conservative
        multirocket = MultiRocket(n_kernels=500, n_jobs=1, random_state=42)
        hydra = HydraTransformer(n_kernels=2, n_groups=16, random_state=42)
    elif size_mb > 1000:
        # Large datasets - conservative
        multirocket = MultiRocket(n_kernels=1000, n_jobs=2, random_state=42)
        hydra = HydraTransformer(n_kernels=4, n_groups=32, random_state=42)
    elif size_mb > 500:
        # Medium datasets - balanced
        multirocket = MultiRocket(n_kernels=2000, n_jobs=3, random_state=42)
        hydra = HydraTransformer(n_kernels=8, n_groups=64, random_state=42)
    else:
        # Small datasets - can be more aggressive
        multirocket = MultiRocket(n_kernels=5000, n_jobs=4, random_state=42)
        hydra = HydraTransformer(n_kernels=16, n_groups=128, random_state=42)

    scaler_std, scaler_hydra = StandardScaler(), StandardScaler()
    
    # Process one transformer at a time to reduce peak memory
    Xt_hydra = scaler_hydra.fit_transform(hydra.fit_transform(X_sample))
    memory_safe_cleanup()
    
    Xt_multi = scaler_std.fit_transform(multirocket.fit_transform(X_sample))
    memory_safe_cleanup()
    
    features = np.concatenate([Xt_hydra, Xt_multi], axis=1)

    return multirocket, hydra, scaler_std, scaler_hydra, features


def process_large_data_in_chunks(X_data, transformer, scaler, chunk_size=1000):
    """Process large datasets in chunks to avoid memory spikes."""
    if len(X_data) <= chunk_size:
        return scaler.transform(transformer.transform(X_data))
    
    results = []
    for i in range(0, len(X_data), chunk_size):
        chunk = X_data[i:i + chunk_size]
        transformed_chunk = scaler.transform(transformer.transform(chunk))
        results.append(transformed_chunk)
        memory_safe_cleanup()
    
    return np.vstack(results)


def process_dataset_resample(args):
    """
    Memory-optimized version for parallel execution.
    """
    folder, dataset, resample_id, model_dir = args
    
    # Check memory before starting
    if memory_usage_mb() > Config.max_memory_mb * 0.9:
        print(f"⚠️  High memory before {dataset}, waiting...")
        memory_safe_cleanup()
    
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
        sample_size = min(50, len(X_train))  # Reduced sample size for memory
        multirocket, hydra, scaler_std, scaler_hydra, X_fit = fit_transformers_memory_safe(
            X_train[:sample_size], n_features, train_size
        )
        
        # Use simpler classifier for initial fit
        clf = RidgeClassifier(alpha=1.0, solver='sparse_cg')  # Memory-efficient solver
        clf.fit(X_fit, y_train[:sample_size])
        memory_safe_cleanup()

        # Transform training data with chunking if large
        if len(X_train) > 1000 and Config.chunk_processing:
            Xt_hydra = process_large_data_in_chunks(X_train, hydra, scaler_hydra)
            memory_safe_cleanup()
            Xt_multi = process_large_data_in_chunks(X_train, multirocket, scaler_std)
            memory_safe_cleanup()
        else:
            Xt_hydra = scaler_hydra.transform(hydra.transform(X_train))
            memory_safe_cleanup()
            Xt_multi = scaler_std.transform(multirocket.transform(X_train))
            memory_safe_cleanup()
        
        X_train_features = np.concatenate([Xt_hydra, Xt_multi], axis=1)
        
        # Fit final classifier
        clf.fit(X_train_features, y_train)
        memory_safe_cleanup()

        # Transform test data
        if len(X_test) > 1000 and Config.chunk_processing:
            Xt_hydra_test = process_large_data_in_chunks(X_test, hydra, scaler_hydra)
            Xt_multi_test = process_large_data_in_chunks(X_test, multirocket, scaler_std)
        else:
            Xt_hydra_test = scaler_hydra.transform(hydra.transform(X_test))
            Xt_multi_test = scaler_std.transform(multirocket.transform(X_test))
        
        X_test_features = np.concatenate([Xt_hydra_test, Xt_multi_test], axis=1)
        acc = accuracy_score(y_test, clf.predict(X_test_features))

        # Save model with compression
        joblib.dump({
            "clf": clf,
            "hydra": hydra,
            "multirocket": multirocket,
            "scaler_std": scaler_std,
            "scaler_hydra": scaler_hydra
        }, model_file, compress=3)

        memory_safe_cleanup()
        return (dataset, resample_id, acc)
    
    except Exception as e:
        print(f"❌ Error processing {dataset} resample {resample_id}: {str(e)}")
        memory_safe_cleanup()
        return (dataset, resample_id, np.nan)


def evaluate_all_memory_safe(folder, output_file=Config.output_file, model_dir=Config.model_dir, 
                            n_resamples=Config.n_resamples, max_workers=Config.max_workers):
    """
    Memory-safe parallel evaluation with batching and monitoring.
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

    # Sort datasets by size (smallest first)
    datasets.sort(key=lambda d: sum(dataset_info(folder, d)[:2]))

    print(f"🚀 Processing {len(datasets)} datasets with {max_workers} workers (memory-safe)")
    print(f"💾 Memory limit: {Config.max_memory_mb}MB | Batch size: {Config.batch_size}")
    print(f"📊 Current memory: {memory_usage_mb():.0f}MB")

    # Process datasets in smaller batches to control memory
    batch_size = Config.batch_size
    all_tasks = []
    
    for dataset in datasets:
        for r in range(1, n_resamples + 1):
            resample_model_dir = os.path.join(model_dir, f"resample_{r}")
            all_tasks.append((folder, dataset, r, resample_model_dir))

    print(f"📋 Total tasks: {len(all_tasks)} | Processing in batches of {batch_size * n_resamples}")

    # Process in batches
    completed_tasks = 0
    total_tasks = len(all_tasks)
    
    for batch_start in range(0, total_tasks, batch_size * n_resamples):
        batch_end = min(batch_start + (batch_size * n_resamples), total_tasks)
        current_batch = all_tasks[batch_start:batch_end]
        
        batch_datasets = set(task[1] for task in current_batch)
        print(f"\n🔄 Processing batch: {', '.join(batch_datasets)}")
        print(f"📦 Batch tasks: {len(current_batch)} | Memory: {memory_usage_mb():.0f}MB")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(process_dataset_resample, task): task for task in current_batch}
            
            for future in as_completed(future_to_task):
                dataset, resample_id, acc = future.result()
                completed_tasks += 1
                
                # Update results
                if dataset in df["Dataset"].values:
                    df.loc[df["Dataset"] == dataset, f"Resample_{resample_id}"] = acc
                    resample_cols = [f"Resample_{i}" for i in range(1, n_resamples + 1)]
                    current_accs = df.loc[df["Dataset"] == dataset, resample_cols].values[0]
                    df.loc[df["Dataset"] == dataset, "MeanAccuracy"] = np.nanmean(current_accs)
                else:
                    row = {"Dataset": dataset, **{f"Resample_{i}": np.nan for i in range(1, n_resamples + 1)}}
                    row[f"Resample_{resample_id}"] = acc
                    row["MeanAccuracy"] = acc if acc is not None else np.nan
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                
                status = f"✅ {acc:.4f}" if acc is not None else "⚠️  Exists"
                print(f"   {dataset} R{resample_id}: {status} [{completed_tasks}/{total_tasks}]")
                
                # Save progress
                if completed_tasks % 5 == 0:
                    df.to_csv(output_file, index=False)
                    if Config.memory_monitor:
                        print(f"💾 Memory check: {memory_usage_mb():.0f}MB")
        
        # Force cleanup between batches
        print("🧹 Cleaning up between batches...")
        memory_safe_cleanup()
        
        # Save batch results
        df.to_csv(output_file, index=False)
    
    # Final save and summary
    final_mean = df["MeanAccuracy"].dropna().mean() if len(df) else None
    print("\n" + "="*60)
    print("✅ All datasets complete!")
    if final_mean is not None:
        print(f"🏆 Overall Mean Accuracy: {final_mean:.4f}")
    else:
        print("⚠️ No valid results")
    print(f"💾 Peak memory usage: {memory_usage_mb():.0f}MB")
    print("="*60)
    
    return df, final_mean


def evaluate_all(folder, output_file=Config.output_file, model_dir=Config.model_dir, 
                 n_resamples=Config.n_resamples):
    """
    Main evaluation function - uses memory-safe parallel processing.
    """
    return evaluate_all_memory_safe(folder, output_file, model_dir, n_resamples, Config.max_workers)


if __name__ == "__main__":
    pass