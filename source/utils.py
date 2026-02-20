

import os
import argparse
import pandas as pd

def sorting_key(id_value):
    if id_value.startswith('C'):    # make harder to find
        return (2, id_value)
    elif id_value.startswith('D'):
        return (0, id_value)
    else:
        return (1, id_value)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--seed_cutoff', type=float, required=True, default=0.5)
    parser.add_argument('--network_output_dir', type=str, required=True)
    parser.add_argument('--submine_data_dir', type=str, required=True)
    parser.add_argument('--edge_cutoff', type=float, default=0.85)

    args = parser.parse_args()
    target = args.target
    threshold = str(args.threshold)
    seed_cutoff = str(args.seed_cutoff)

    output_dir = os.path.abspath(args.network_output_dir)
    data_dir = os.path.join(os.path.abspath(args.submine_data_dir), threshold, target)

    sNPresult_path = os.path.join(output_dir, f"NP_SUBDYVE_{args.seed_cutoff}_{args.edge_cutoff}_subgraph.txt")
    sNPSeed_path = os.path.join(output_dir, f"seed_{seed_cutoff}.txt")
    output_final_path = os.path.join(output_dir, "NP_SUBDYVE_result_excludeseed.txt")

    try:
        print(f"Post-processing {target}")

        np_df = pd.read_csv(sNPresult_path, sep='\t', header=None, names=['id', 'npscore'])
        np_df.sort_values(by='npscore', ascending=False, inplace=True)
        np_df.reset_index(drop=True, inplace=True)

        seed_df = pd.read_csv(sNPSeed_path, sep='\t', header=None, names=['id', 'weight'])
        seed_ids = seed_df['id'].tolist()

        np_excl = np_df[~np_df['id'].isin(seed_ids)].reset_index(drop=True)

        np_excl = np_excl[~np_excl['id'].str.startswith('t')].reset_index(drop=True)

        active_ids = pd.read_csv(os.path.join(data_dir, "ACTIVE_DATA.csv"))['CHEMBLID'].tolist()
        decoy_ids = pd.read_csv(os.path.join(data_dir, "DECOY_DATA.csv"))['DECOYID'].tolist()

        existing_ids = set(np_excl['id'])
        missing = [
            {'id': i, 'npscore': 0.0} for i in active_ids + decoy_ids if i not in existing_ids
        ]
        np_full = pd.concat([np_excl, pd.DataFrame(missing)], ignore_index=True)

        np_full_sorted = np_full.sort_values(
            by=['npscore', 'id'],
            ascending=[False, True],
            key=lambda x: x if x.name == 'npscore' else x.map(sorting_key)
        ).reset_index(drop=True)

        np_full_sorted.to_csv(output_final_path, sep='\t', index=False)
        print(f"Saved: {output_final_path}")

    except Exception as e:
        print(f"Error during post-processing for {target}: {e}")

if __name__ == "__main__":
    main()
