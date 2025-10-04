import os
import gc
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
    """Optimized configuration for 50GB RAM, 9 threads"""
    n_resamples = 30
    output_file = "ucr_baseline_results_wide.csv"
    model_dir = "baseline_models_resamples"
    n_jobs_per_dataset = 2  # Threads per dataset
    parallel_datasets = 4  # Process 4 datasets simultaneously
    checkpoint_interval = 3  # Save every 3 resamples


def get_transformer_params(n_samples, n_features):
    """Smart parameter selection based on dataset characteristics."""
    if n_samples > 5000 or n_features > 1000:
        return {
            'mr': {'n_kernels': 5000, 'n_jobs': Config.n_jobs_per_dataset},
            'hydra': {'n_kernels': 4, 'n_groups': 32}
        }
    elif n_samples > 1000:
        return {
            'mr': {'n_kernels': 8000, 'n_jobs': Config.n_jobs_per_dataset},
            'hydra': {'n_kernels': 6, 'n_groups': 48}
        }
    else:
        return {
            'mr': {'n_kernels': 10000, 'n_jobs': Config.n_jobs_per_dataset},
            'hydra': {'n_kernels': 8, 'n_groups': 64}
        }


def load_and_resample(train_file, test_file, resample_id):
    """Load, combine, shuffle, and split data in one pass."""
    train_df = pd.read_csv(train_file, sep="\t", header=None)
    test_df = pd.read_csv(test_file, sep="\t", header=None)
    
    n_train = len(train_df)
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=resample_id).reset_index(drop=True)
    
    train_resampled = full_df.iloc[:n_train]
    test_resampled = full_df.iloc[n_train:]
    
    # Extract and reshape in one step
    y_train = train_resampled.iloc[:, 0].values
    X_train = train_resampled.iloc[:, 1:].values[:, np.newaxis, :]
    y_test = test_resampled.iloc[:, 0].values
    X_test = test_resampled.iloc[:, 1:].values[:, np.newaxis, :]
    
    return X_train, y_train, X_test, y_test


def process_resample(folder, dataset, resample_id, model_dir):
    """Process a single resample: train and evaluate."""
    np.random.seed(resample_id)
    
    # Check if already computed
    resample_dir = os.path.join(model_dir, f"resample_{resample_id}", dataset)
    model_file = os.path.join(resample_dir, "clf.pkl")
    
    if os.path.exists(model_file):
        # Load existing accuracy
        try:
            model_data = joblib.load(model_file)
            if 'accuracy' in model_data:
                return resample_id, model_data['accuracy'], 'loaded'
        except:
            pass  # Recompute if loading fails
    
    # Load data
    train_file = os.path.join(folder, dataset, f"{dataset}_TRAIN.tsv")
    test_file = os.path.join(folder, dataset, f"{dataset}_TEST.tsv")
    X_train, y_train, X_test, y_test = load_and_resample(train_file, test_file, resample_id)
    
    # Get optimal parameters
    params = get_transformer_params(X_train.shape[0], X_train.shape[2])
    
    # Initialize transformers
    multirocket = MultiRocket(random_state=42, **params['mr'])
    hydra = HydraTransformer(random_state=42, **params['hydra'])
    
    # Fit on sample for speed
    sample_size = min(100, len(X_train))
    X_sample = X_train[:sample_size]
    y_sample = y_train[:sample_size]
    
    # Transform pipeline
    hydra_features = hydra.fit_transform(X_sample)
    mr_features = multirocket.fit_transform(X_sample)
    
    scaler_hydra = StandardScaler().fit(hydra_features)
    scaler_mr = StandardScaler().fit(mr_features)
    
    X_sample_scaled = np.hstack([
        scaler_hydra.transform(hydra_features),
        scaler_mr.transform(mr_features)
    ])
    
    # Train classifier
    clf = RidgeClassifier(random_state=42).fit(X_sample_scaled, y_sample)
    
    # Full training
    X_train_hydra = scaler_hydra.transform(hydra.transform(X_train))
    X_train_mr = scaler_mr.transform(multirocket.transform(X_train))
    X_train_full = np.hstack([X_train_hydra, X_train_mr])
    clf.fit(X_train_full, y_train)
    
    # Evaluate
    X_test_hydra = scaler_hydra.transform(hydra.transform(X_test))
    X_test_mr = scaler_mr.transform(multirocket.transform(X_test))
    X_test_full = np.hstack([X_test_hydra, X_test_mr])
    
    y_pred = clf.predict(X_test_full)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model with accuracy
    os.makedirs(resample_dir, exist_ok=True)
    joblib.dump({
        'clf': clf,
        'hydra': hydra,
        'multirocket': multirocket,
        'scaler_hydra': scaler_hydra,
        'scaler_mr': scaler_mr,
        'accuracy': accuracy
    }, model_file)
    
    # Cleanup
    del X_train, X_test, X_train_full, X_test_full
    gc.collect()
    
    return resample_id, accuracy, 'computed'


def process_dataset_resamples(args):
    """Process all missing resamples for a single dataset."""
    folder, dataset, missing_resamples, model_dir = args
    
    results = {}
    for r in missing_resamples:
        resample_id, acc, status = process_resample(folder, dataset, r, model_dir)
        results[resample_id] = (acc, status)
    
    return dataset, results


def check_and_compute_missing(df, folder, model_dir, n_resamples):
    """Check every single resample for every dataset and compute missing ones."""
    resample_cols = [f"Resample_{i}" for i in range(1, n_resamples + 1)]
    
    # Get all datasets from folder
    all_datasets = sorted([d for d in os.listdir(folder) 
                          if os.path.isdir(os.path.join(folder, d))])
    
    # Add missing datasets to dataframe
    existing_datasets = set(df['Dataset'].tolist()) if len(df) > 0 else set()
    new_datasets = set(all_datasets) - existing_datasets
    
    if new_datasets:
        print(f"📝 Adding {len(new_datasets)} new datasets to tracking")
        for dataset in new_datasets:
            new_row = {'Dataset': dataset}
            for col in resample_cols:
                new_row[col] = np.nan
            new_row['MeanAccuracy'] = np.nan
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Now check EVERY resample for EVERY dataset
    total_missing = 0
    dataset_status = {}
    
    print(f"\n🔍 Scanning all {len(df)} datasets × {n_resamples} resamples...")
    
    for idx, row in df.iterrows():
        dataset = row['Dataset']
        missing_resamples = []
        
        for r in range(1, n_resamples + 1):
            col = f"Resample_{r}"
            if pd.isna(row[col]):
                missing_resamples.append(r)
                total_missing += 1
        
        dataset_status[dataset] = missing_resamples
        
        if missing_resamples:
            print(f"   ⚠️  {dataset}: Missing {len(missing_resamples)} resamples {missing_resamples[:5]}{'...' if len(missing_resamples) > 5 else ''}")
        else:
            print(f"   ✅ {dataset}: Complete ({n_resamples}/{n_resamples})")
    
    if total_missing == 0:
        print(f"\n✨ All datasets complete! Total: {len(df)} datasets × {n_resamples} resamples")
        return df, dataset_status
    
    print(f"\n🔧 Found {total_missing} missing resamples across {len([d for d, m in dataset_status.items() if m])} datasets")
    print(f"⚡ Starting parallel computation ({Config.parallel_datasets} datasets at a time)...\n")
    
    return df, dataset_status


def evaluate_all(folder, output_file=Config.output_file, model_dir=Config.model_dir, 
                 n_resamples=Config.n_resamples):
    """Optimized evaluation with parallel dataset processing."""
    
    resample_cols = [f"Resample_{i}" for i in range(1, n_resamples + 1)]
    
    # Load or initialize dataframe
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        print(f"📂 Loaded existing results: {len(df)} datasets")
    else:
        df = pd.DataFrame(columns=['Dataset'] + resample_cols + ['MeanAccuracy'])
        print(f"📂 Creating new results file")
    
    # Check and identify ALL missing resamples
    df, dataset_status = check_and_compute_missing(df, folder, model_dir, n_resamples)
    
    # Filter to only datasets with missing resamples
    datasets_to_process = [(d, dataset_status[d]) for d in dataset_status if dataset_status[d]]
    
    if not datasets_to_process:
        print("\n🎉 Nothing to compute - all datasets complete!")
        return df
    
    print(f"\n{'='*70}")
    print(f"🚀 Processing {len(datasets_to_process)} datasets with missing resamples")
    print(f"⚡ Parallel processing: {Config.parallel_datasets} datasets × {Config.n_jobs_per_dataset} threads = {Config.parallel_datasets * Config.n_jobs_per_dataset} total threads")
    print(f"{'='*70}\n")
    
    # Prepare arguments for parallel processing
    process_args = [
        (folder, dataset, missing_resamples, model_dir)
        for dataset, missing_resamples in datasets_to_process
    ]
    
    # Process datasets in parallel batches
    completed = 0
    total = len(datasets_to_process)
    
    with ProcessPoolExecutor(max_workers=Config.parallel_datasets) as executor:
        # Submit all tasks
        future_to_dataset = {
            executor.submit(process_dataset_resamples, args): args[1] 
            for args in process_args
        }
        
        # Process results as they complete
        for future in as_completed(future_to_dataset):
            dataset = future_to_dataset[future]
            completed += 1
            
            try:
                dataset, results = future.result()
                
                print(f"[{completed}/{total}] ✅ {dataset} completed!")
                
                # Update dataframe with results
                for resample_id, (acc, status) in results.items():
                    col = f"Resample_{resample_id}"
                    df.loc[df['Dataset'] == dataset, col] = acc
                    
                    symbol = "📁" if status == 'loaded' else "⚙️"
                    print(f"   {symbol} Resample {resample_id}: {acc:.4f} ({status})")
                
                # Calculate mean accuracy
                row = df[df['Dataset'] == dataset].iloc[0]
                accuracies = [row[c] for c in resample_cols if not pd.isna(row[c])]
                mean_acc = np.mean(accuracies) if accuracies else np.nan
                df.loc[df['Dataset'] == dataset, 'MeanAccuracy'] = mean_acc
                
                print(f"   📈 Mean accuracy: {mean_acc:.4f}\n")
                
                # Save after each dataset completion
                df.to_csv(output_file, index=False)
                
            except Exception as e:
                print(f"[{completed}/{total}] ❌ {dataset} failed: {e}\n")
            
            gc.collect()
    
    # Final save and statistics
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print("✅ EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"📊 Total datasets: {len(df)}")
    print(f"🎯 Datasets processed: {len(datasets_to_process)}")
    print(f"🔢 Total resamples computed: {sum(len(m) for _, m in datasets_to_process)}")
    
    complete_datasets = df['MeanAccuracy'].notna().sum()
    print(f"✓  Complete datasets: {complete_datasets}/{len(df)}")
    
    if complete_datasets > 0:
        print(f"🏆 Overall mean accuracy: {df['MeanAccuracy'].mean():.4f}")
    
    print(f"💾 Results saved to: {output_file}")
    final_mean = df["MeanAccuracy"].dropna().mean() if len(df) else None
    return df, final_mean


if __name__ == "__main__":
    pass