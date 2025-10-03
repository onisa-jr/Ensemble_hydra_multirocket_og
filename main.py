import os
import numpy as np
import pandas as pd
from ridgenet import train_fld
from hydra_mr_resample import evaluate_all, Config

DATASET_FOLDER = "datasets"
OUTPUT_FILE = "ridgenet.csv"
N_RESAMPLES = 30


def load_existing_results(output_file: str, n_resamples: int):
    """Load existing results if available, else create a new DataFrame."""
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        processed = set(df["Dataset"].tolist())
        print(f"📂 Resuming from existing results: {len(processed)} datasets already done")
    else:
        cols = ["Dataset"] + [f"Resample{i}" for i in range(1, n_resamples + 1)] + ["MeanAccuracy"]
        df, processed = pd.DataFrame(columns=cols), set()
        print(f"📂 Starting fresh run, output will be saved to {output_file}")
    return df, processed


def update_results(df: pd.DataFrame, dataset: str, resample_id: int, acc: float, res_accs: list):
    """Update results DataFrame live after each resample."""
    # If dataset row exists → update it
    if dataset in df["Dataset"].values:
        df.loc[df["Dataset"] == dataset, f"Resample{resample_id}"] = acc
        df.loc[df["Dataset"] == dataset, "MeanAccuracy"] = np.nanmean(res_accs)
    else:
        row = {"Dataset": dataset,
               **{f"Resample{i+1}": np.nan for i in range(len(res_accs))},
               "MeanAccuracy": np.nan}
        row[f"Resample{resample_id}"] = acc
        row["MeanAccuracy"] = np.nanmean(res_accs)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def run_experiments(dataset_folder: str, output_file: str, n_resamples: int):
    """Main loop to train RidgeNet across datasets and resamples."""
    evaluate_all(dataset_folder) 

    df, processed = load_existing_results(output_file, n_resamples)

    for dataset in os.listdir(dataset_folder):
        dataset_path = os.path.join(dataset_folder, dataset)
        if not os.path.isdir(dataset_path):
            continue  # Skip non-folder entries

        if dataset in processed:
            print(f"⏭️  Skipping {dataset} (already processed)")
            continue

        print(f"\n🚀 Running RidgeNet on {dataset}")
        res_accs = []
        for resample_id in range(1, n_resamples + 1):
            acc = train_fld(DATASET=dataset_path, RESAMPLE=resample_id, BATCH_SIZE=64)
            res_accs.append(acc if acc is not None else np.nan)

            if acc is not None:
                print(f"   Resample {resample_id}: Accuracy = {acc:.4f}")
            else:
                print(f"   Resample {resample_id}: Skipped (model already exists)")

            # Live update results
            df = update_results(df, dataset, resample_id, res_accs[-1], res_accs)
            df.to_csv(output_file, index=False)

        print(f"✅ Finished {dataset} | Mean Accuracy: {np.nanmean(res_accs):.4f}")

    print("\n🎉 All datasets processed.")
    return df


if __name__ == "__main__":
    run_experiments(DATASET_FOLDER, OUTPUT_FILE, N_RESAMPLES)
