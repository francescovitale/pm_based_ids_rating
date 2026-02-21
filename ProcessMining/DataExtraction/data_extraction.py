import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FP_DIR = os.path.join('Input', 'FP')
INPUT_TP_DIR = os.path.join('Input', 'TP')

OUTPUT_BASE_DIR = 'Output'
CSV_FILE = 'false_positives.csv'

REPETITIONS = 5
TRAINING_SIZE = 0.5
RANDOM_SEED_BASE = 42

CLASSIFIERS = [
    'ocsvm',
    'autoencoder',
    'variational_autoencoder'
]

def copy_files(file_list, destination):
    os.makedirs(destination, exist_ok=True)
    for src in file_list:
        try:
            shutil.copy2(src, destination)
        except Exception as e:
            print(f"Error copying {src}: {e}")

def main():
    print("\n--- Starting Data Split Process (FP + TP) ---")

    if not os.path.exists(CSV_FILE):
        print(f"ERROR: CSV file not found: {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)

    fp_file_map = {}
    for _, row in df.iterrows():
        pcap_name = f"{row['ID']}.pcap"
        full_path = os.path.join(INPUT_FP_DIR, pcap_name)

        if os.path.exists(full_path):
            fp_file_map[full_path] = row

    print(f"Loaded {len(fp_file_map)} False Positive files")

    tp_files = []
    if os.path.exists(INPUT_TP_DIR):
        tp_files = [
            os.path.join(INPUT_TP_DIR, f)
            for f in os.listdir(INPUT_TP_DIR)
            if f.endswith('.pcap')
        ]

    print(f"Loaded {len(tp_files)} True Positive files")

    for r in range(REPETITIONS):
        rep_name = f"Repetition_{r + 1}"
        seed = RANDOM_SEED_BASE + r

        print(f"\nProcessing {rep_name}")

        for clf in CLASSIFIERS:
            print(f"  Classifier: {clf}")

            fp_list = [
                path for path, row in fp_file_map.items()
                if row.get(clf, 0) == 1
            ]

            if len(fp_list) > 1:
                fp_train, fp_infer = train_test_split(
                    fp_list,
                    train_size=TRAINING_SIZE,
                    random_state=seed,
                    shuffle=True
                )
            elif len(fp_list) == 1:
                fp_train = fp_list
                fp_infer = []
            else:
                fp_train, fp_infer = [], []

            base_path = os.path.join(OUTPUT_BASE_DIR, rep_name, clf)

            copy_files(
                fp_train,
                os.path.join(base_path, 'Training', 'FP')
            )

            copy_files(
                fp_infer,
                os.path.join(base_path, 'Inference', 'FP')
            )

            copy_files(
                tp_files,
                os.path.join(base_path, 'Inference', 'TP')
            )

            print(
                f"    FP → Training: {len(fp_train)} | "
                f"Inference: {len(fp_infer)} | "
                f"TP → Inference: {len(tp_files)}"
            )

    print("\n--- Processing Complete ---")
    print("False Positives were split into Training/Inference.")
    print("All True Positives were added to Inference.")

if __name__ == "__main__":
    main()
