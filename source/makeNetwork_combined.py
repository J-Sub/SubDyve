import os
import pickle
import time
import itertools
import argparse
import pandas as pd
import math
import statistic

def compute_similarity(fp_dict, iteration, sim_type, cross_fp=None):
    if cross_fp is None:
        return {
            "subgraph": statistic.ParallelCalcTanimotoForFPs(
                (fp_dict["subgraph"], iteration, 'subgraph', sim_type)
            )
        }
    else:
        return {
            "subgraph": statistic.ParallelCalcTanimotoForTwoFPs(
                (fp_dict["subgraph"], cross_fp["subgraph"], iteration, 'subgraph', sim_type)
            )
        }

def save_similarity(out_path, name, data):
    file = os.path.join(out_path, f"Sim_{name}.pickle")
    with open(file, "wb") as f:
        pickle.dump(data, f)
    return file

def make_seed_file(dfSeed, seed_cutoff, out_dir):
    path = os.path.join(out_dir, f"seed_{seed_cutoff}.txt")
    dLogCut = math.log10(seed_cutoff)
    nPos = 0
    with open(path, "w") as f:
        for idx, row in dfSeed.iterrows():
            dacvalue = math.log10(row['acvalue'])
            if dacvalue < seed_cutoff:
                dfSeed.loc[idx, 'id'] = row['id'].replace("t", "p")
                f.write(dfSeed.loc[idx, 'id'] + "\t1\n")
                nPos += 1
    print(f"Seed file written ({nPos} positive): {path}")
    return path

def make_edge_file(dfA, dfS, dfD, sims, edge_cut, out_dir, seed_cutoff):
    idmap = lambda df: {str(i): v for i, v in enumerate(df['id'].tolist())}
    dictA, dictS, dictD = idmap(dfA), idmap(dfS), idmap(dfD)
    sType = 'subgraph'
    nSeed = len(sims["seed"][sType])

    out_path = os.path.join(out_dir, f"edge_{seed_cutoff}_{edge_cut}_{sType}.txt")
    with open(out_path, "w") as f:

        for line in sims["seed"][sType]:
            i1, i2, sim = line.strip().split('\t')
            f.write(f"{dictS[i1]}\t{dictS[i2]}\t{sim}\n")

        for line in sims["active"][sType]:
            i1, i2, sim = line.strip().split('\t')
            f.write(f"{dictA[i1]}\t{dictA[i2]}\t{sim}\n")

        for line in sims["decoy"][sType]:
            i1, i2, sim = line.strip().split('\t')
            f.write(f"{dictD[i1]}\t{dictD[i2]}\t{sim}\n")

        def write_cross(sim_list, dict1, dict2, label):
            buckets = [[] for _ in range(101)]
            for line in sim_list[sType]:
                i1, i2, sim = line.strip().split('\t')
                b = min(int(float(sim) * 100), 100)
                buckets[b].append(f"{dict1[i1]}\t{dict2[i2]}\t{sim}\n")
            count = 0
            for i in reversed(range(101)):
                if i >= 85 or (count < nSeed and i >= 40):
                    for l in buckets[i]:
                        f.write(l)
                        count += 1
                else:
                    break

        write_cross(sims["active_seed"], dictA, dictS, "A-S")
        write_cross(sims["seed_decoy"], dictS, dictD, "S-D")
        write_cross(sims["active_decoy"], dictA, dictD, "A-D")

    print(f"Edge file written: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--active", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--decoy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compound_cutoff", type=float, default=0.9)
    parser.add_argument("--seed_cutoff", type=float, default=0.5)
    parser.add_argument("--edge_cutoff", type=float, default=0.85)
    parser.add_argument("--similarity", default='tanimoto')
    parser.add_argument("--input_fp_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    dfA, dfS, dfD = statistic.removeRedundancy(args.active, args.seed, args.decoy)
    fps = statistic.load_existing_fps(args.input_fp_dir, args.compound_cutoff)
    sims = {
        "active": compute_similarity(fps["active"], 0.0, args.similarity),
        "seed": compute_similarity(fps["seed"], 0.85, args.similarity),
        "decoy": compute_similarity(fps["decoy"], 0.85, args.similarity),
        "active_seed": compute_similarity(fps["active"], 0.3, args.similarity, cross_fp=fps["seed"]),
        "seed_decoy": compute_similarity(fps["seed"], 0.3, args.similarity, cross_fp=fps["decoy"]),
        "active_decoy": compute_similarity(fps["active"], 0.3, args.similarity, cross_fp=fps["decoy"]),
    }

    for k, v in sims.items():
        save_similarity(args.output, k.capitalize(), v)
    seed_path = make_seed_file(dfS, args.seed_cutoff, args.output)
    make_edge_file(dfA, dfS, dfD, sims, args.edge_cutoff, args.output, args.seed_cutoff)
    print("All processing complete.")


if __name__ == "__main__":
    main()
