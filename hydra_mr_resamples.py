import os
import gc
import psutil
import numpy as np
import pandas as pd
import joblib
import warnings
from tqdm.auto import tqdm
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from aeon.transformations.collection.convolution_based import MultiRocket, HydraTransformer
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path

warnings.filterwarnings("ignore")


class Config:
    """High-performance configuration"""
    n_resamples = 30
    max_memory_mb = 45000  # Increased to match your 45GB RAM
    skip_large_mb = 5000   # Increased threshold for large datasets
    skip_large_features = 20000  # Increased feature threshold
    output_file = "ucr_baseline_results_wide.csv"
    model_dir = "baseline_models_resamples"
    n_jobs = 9  # Utilize all 9 threads
    chunk_size = 10000  # Process data in chunks for memory efficiency


def memory_usage_mb():
    """Return current memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def aggressive_cleanup():
    """Aggressive garbage collection with memory pre-allocation hints."""
    gc.collect()
    if hasattr(gc, 'freeze'):  # Python 3.10+
        gc.freeze()
    gc.collect()


def dataset_info(folder, name):
    """Return (train_size_mb, test_size_mb, n_features) with caching."""
    train_file = os.path.join(folder, name, f"{name}_TRAIN.tsv")
    test_file = os.path.join(folder, name, f"{name}_TEST.tsv")

    train_size = os.path.getsize(train_file) / (1024 * 1024) if os.path.exists(train_file) else 0
    test_size = os.path.getsize(test_file) / (1024 * 1024) if os.path.exists(test_file) else 0

    if train_size > 0:
        # Use numpy for faster reading of just the first row
        sample = np.loadtxt(train_file, delimiter='\t', max_rows=1)
        n_features = sample.shape[0] - 1
    else:
        n_features = 0

    return train_size, test_size, n_features


def optimize_data_loading(file_path, n_rows=None):
    """Optimized data loading using numpy for better performance."""
    data = np.loadtxt(file_path, delimiter='\t', max_rows=n_rows)
    return data


def fit_transformers_optimized(X_sample, n_features, size_mb, dataset_name):
    """Optimized transformer fitting with memory-aware parameters."""
    
    # Memory-aware parameter tuning based on dataset size and available resources
    if size_mb > 2000:
        # Large datasets - conservative settings
        multirocket = MultiRocket(n_kernels=800, n_jobs=2, random_state=42)
        hydra = HydraTransformer(n_kernels=2, n_groups=4, random_state=42)
    elif size_mb > 500:
        # Medium datasets - balanced settings
        multirocket = MultiRocket(n_kernels=2000, n_jobs=4, random_state=42)
        hydra = HydraTransformer(n_kernels=4, n_groups=8, random_state=42)
    else:
        # Small datasets - aggressive settings
        multirocket = MultiRocket(n_kernels=5000, n_jobs=8, random_state=42)
        hydra = HydraTransformer(n_kernels=8, n_groups=16, random_state=42)

    # Use float32 to save memory
    X_sample = X_sample.astype(np.float32)
    
    scaler_hydra = StandardScaler()
    scaler_std = StandardScaler()
    
    # Process transformers in optimized order
    Xt_hydra = scaler_hydra.fit_transform(hydra.fit_transform(X_sample))
    Xt_multi = scaler_std.fit_transform(multirocket.fit_transform(X_sample))
    
    # Concatenate efficiently
    features = np.concatenate([Xt_hydra, Xt_multi], axis=1, dtype=np.float32)
    
    return multirocket, hydra, scaler_std, scaler_hydra, features


def process_dataset_parallel_wrapper(args):
    """Wrapper for parallel processing."""
    return process_dataset_optimized(*args)


def process_dataset_optimized(folder, dataset, resample_id, model_dir):
    """Highly optimized dataset processing."""
    # Set random seed for reproducibility
    np.random.seed(resample_id)

    # Optimized path handling
    dataset_model_dir = Path(model_dir) / f"resample_{resample_id}" / dataset
    model_file = dataset_model_dir / "clf.pkl"
    
    # Skip if already processed
    if model_file.exists():
        return None

    # Create directory efficiently
    dataset_model_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Optimized data loading
        train_file = Path(folder) / dataset / f"{dataset}_TRAIN.tsv"
        test_file = Path(folder) / dataset / f"{dataset}_TEST.tsv"
        
        # Load data with numpy for better performance
        train_data = optimize_data_loading(train_file)
        test_data = optimize_data_loading(test_file)

        # Combine and shuffle efficiently
        full_data = np.vstack([train_data, test_data])
        rng = np.random.default_rng(resample_id)
        rng.shuffle(full_data, axis=0)

        # Split back
        n_train = len(train_data)
        train_data = full_data[:n_train]
        test_data = full_data[n_train:n_train + len(test_data)]

        # Extract features and labels
        y_train = train_data[:, 0].astype(np.int32)
        X_train = train_data[:, 1:].astype(np.float32).reshape(len(train_data), 1, -1)
        y_test = test_data[:, 0].astype(np.int32)
        X_test = test_data[:, 1:].astype(np.float32).reshape(len(test_data), 1, -1)

        # Get dataset info for parameter tuning
        train_size, _, n_features = dataset_info(folder, dataset)
        
        # Use larger sample for better transformer fitting
        sample_size = min(100, len(X_train))
        multirocket, hydra, scaler_std, scaler_hydra, X_fit = fit_transformers_optimized(
            X_train[:sample_size], n_features, train_size, dataset
        )

        # Initialize classifier with optimized parameters
        clf = RidgeClassifier(alpha=1.0, solver='sparse_cg', random_state=42)
        clf.fit(X_fit, y_train[:sample_size])

        # Process full training data in chunks if large
        if len(X_train) > 1000:
            # Chunk processing for memory efficiency
            chunk_size = Config.chunk_size
            X_train_transformed = []
            
            for i in range(0, len(X_train), chunk_size):
                chunk = X_train[i:i + chunk_size]
                Xt_hydra_chunk = scaler_hydra.transform(hydra.transform(chunk))
                Xt_multi_chunk = scaler_std.transform(multirocket.transform(chunk))
                chunk_features = np.concatenate([Xt_hydra_chunk, Xt_multi_chunk], axis=1, dtype=np.float32)
                X_train_transformed.append(chunk_features)
                
            X_train_features = np.vstack(X_train_transformed)
        else:
            # Process all at once for smaller datasets
            Xt_hydra = scaler_hydra.transform(hydra.transform(X_train))
            Xt_multi = scaler_std.transform(multirocket.transform(X_train))
            X_train_features = np.concatenate([Xt_hydra, Xt_multi], axis=1, dtype=np.float32)

        # Fit final classifier
        clf.fit(X_train_features, y_train)

        # Transform test data
        if len(X_test) > 1000:
            X_test_transformed = []
            for i in range(0, len(X_test), chunk_size):
                chunk = X_test[i:i + chunk_size]
                Xt_hydra_chunk = scaler_hydra.transform(hydra.transform(chunk))
                Xt_multi_chunk = scaler_std.transform(multirocket.transform(chunk))
                chunk_features = np.concatenate([Xt_hydra_chunk, Xt_multi_chunk], axis=1, dtype=np.float32)
                X_test_transformed.append(chunk_features)
            X_test_features = np.vstack(X_test_transformed)
        else:
            Xt_hydra = scaler_hydra.transform(hydra.transform(X_test))
            Xt_multi = scaler_std.transform(multirocket.transform(X_test))
            X_test_features = np.concatenate([Xt_hydra, Xt_multi], axis=1, dtype=np.float32)

        # Predict and calculate accuracy
        y_pred = clf.predict(X_test_features)
        acc = accuracy_score(y_test, y_pred)

        # Save model with compression
        joblib.dump({
            "clf": clf,
            "hydra": hydra,
            "multirocket": multirocket,
            "scaler_std": scaler_std,
            "scaler_hydra": scaler_hydra
        }, model_file, compress=3)  # Level 3 compression for speed/size balance

        return acc

    except Exception as e:
        print(f"❌ Error processing {dataset} (resample {resample_id}): {str(e)}")
        return None
    finally:
        aggressive_cleanup()


def process_dataset_batch(datasets_batch, folder, resample_id, model_dir):
    """Process a batch of datasets for a single resample."""
    results = {}
    for dataset in datasets_batch:
        acc = process_dataset_optimized(folder, dataset, resample_id, model_dir)
        results[dataset] = acc
    return results


def evaluate_all_parallel(folder, output_file=Config.output_file, 
                         model_dir=Config.model_dir, n_resamples=Config.n_resamples):
    """Highly parallel evaluation using all available cores."""
    
    # Load or create results table
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        processed = set(df["Dataset"].tolist())
        print(f"📂 Resuming from existing results: {len(processed)} datasets already done")
    else:
        cols = ["Dataset"] + [f"Resample_{i}" for i in range(1, n_resamples + 1)] + ["MeanAccuracy"]
        df, processed = pd.DataFrame(columns=cols), set()
        print(f"📂 Starting fresh run, output will be saved to {output_file}")

    # Get all datasets sorted by size
    datasets = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    datasets.sort(key=lambda d: sum(dataset_info(folder, d)[:2]))
    
    # Filter out already processed datasets
    pending_datasets = [d for d in datasets if d not in processed]
    
    if not pending_datasets:
        print("✅ All datasets already processed!")
        return df, df["MeanAccuracy"].mean() if len(df) else None

    print(f"🚀 Processing {len(pending_datasets)} datasets using {Config.n_jobs} parallel jobs")

    # Process resamples in parallel
    with ProcessPoolExecutor(max_workers=Config.n_jobs) as executor:
        futures = {}
        
        for resample_id in range(1, n_resamples + 1):
            # Split datasets into batches for better load balancing
            batch_size = max(1, len(pending_datasets) // (Config.n_jobs * 2))
            dataset_batches = [pending_datasets[i:i + batch_size] 
                             for i in range(0, len(pending_datasets), batch_size)]
            
            for batch in dataset_batches:
                future = executor.submit(process_dataset_batch, batch, folder, resample_id, model_dir)
                futures[future] = (resample_id, batch)

        # Process completed futures
        with tqdm(total=len(futures), desc="Processing datasets") as pbar:
            for future in as_completed(futures):
                resample_id, batch = futures[future]
                try:
                    batch_results = future.result()
                    
                    for dataset, acc in batch_results.items():
                        if acc is not None:
                            # Update results dataframe
                            if dataset in df["Dataset"].values:
                                df.loc[df["Dataset"] == dataset, f"Resample_{resample_id}"] = acc
                            else:
                                row = {"Dataset": dataset}
                                row.update({f"Resample_{i+1}": np.nan for i in range(n_resamples)})
                                row[f"Resample_{resample_id}"] = acc
                                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                            
                            # Update mean accuracy
                            resample_cols = [f"Resample_{i+1}" for i in range(n_resamples)]
                            current_accs = df.loc[df["Dataset"] == dataset, resample_cols].values[0]
                            mean_acc = np.nanmean(current_accs)
                            df.loc[df["Dataset"] == dataset, "MeanAccuracy"] = mean_acc
                    
                    # Save progress after each batch
                    df.to_csv(output_file, index=False)
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"❌ Error in batch processing: {str(e)}")
                    pbar.update(1)

    # Final cleanup and summary
    aggressive_cleanup()
    
    final_mean = df["MeanAccuracy"].mean() if len(df) else None
    print("\n✅ All datasets complete")
    if final_mean is not None:
        print(f"🏆 Overall Mean Accuracy: {final_mean:.4f}")
    else:
        print("⚠️ No valid results")

    return df, final_mean


if __name__ == "__main__":
    # Example usage
    data_folder = "path/to/your/ucr/data"  # Update this path
    
    print(f"💻 Starting parallel processing with {Config.n_jobs} cores")
    print(f"💾 Available memory: {Config.max_memory_mb} MB")
    
    results_df, overall_mean = evaluate_all_parallel(data_folder)
    
    print(f"📊 Final results saved to {Config.output_file}")
    print(f"🎯 Overall performance: {overall_mean:.4f}")