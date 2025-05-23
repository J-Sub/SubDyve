import pandas as pd
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Collect best AUC iteration for each target")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold value used in file paths")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the resulting CSV")
    parser.add_argument("--targets", nargs="+", default=[
        "ACES", "ADA", "ANDR", "EGFR", "FA10",
        "KIT", "PLK1", "SRC", "THRB", "UROK"
    ], help="List of targets to process")
    return parser.parse_args()

def main():
    args = parse_args()
    threshold = args.threshold
    output_path = args.output_path
    targets = args.targets

    target_max_AUC_dict = {}

    for target in targets:
        try:
            path = f"./SSM/results/DUDE/{threshold}/{target}/performance.csv"
            df = pd.read_csv(path)
            max_iter = df.loc[df["AUC"].idxmax(), "iteration"]
            target_max_AUC_dict[target] = int(max_iter)
        except Exception as e:
            print(f"Failed to load performance for {target}: {e}")
            continue

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(target_max_AUC_dict.items(), columns=["target", "max_AUC_iteration"]).to_csv(output_path, index=False)
    print(f"Best AUC iteration saved to {output_path}")

if __name__ == "__main__":
    main()
