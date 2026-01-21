# filename: src/test_compassion_prefs.py
import os
import argparse
import random
import json
import math
import time
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm
from openai import OpenAI

# ========== CLIENT FACTORIES ==========

def mk_client(model_type: str, model_path: str, api_key: str,
              temperature: float = 1.0,
              ignore_proxy: bool = False,
              max_retries: int = 5):
    """
    Create a client wrapper with a unified .invoke(messages) API.
    messages: [{"role": "system"/"user"/"assistant", "content": "..."}]
    """

    if model_type == "openai":
        client = OpenAI(api_key=api_key)

        class Wrapper:
            def invoke(self, messages):
                resp = client.chat.completions.create(
                    model=model_path,
                    messages=messages,
                    temperature=temperature,
                )
                return resp.choices[0].message.content
        return Wrapper()

    elif model_type == "deepseek":
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        class Wrapper:
            def invoke(self, messages):
                resp = client.chat.completions.create(
                    model=model_path,
                    messages=messages,
                    temperature=temperature,
                    stream=False
                )
                return resp.choices[0].message.content
        return Wrapper()

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# ========== HELPERS ==========

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def compute_entropy(shares):
    """ Shannon entropy in bits """
    return -sum(p * math.log(p, 2) for p in shares if p > 0)

def compute_individualism_index(pA, pB, pC, pD):
    return pA - (pB + pC + pD) / 3

def append_row_csv(path, row_dict):
    """安全地把一行追加到 CSV（首写入带表头，其它不带）。"""
    exists = os.path.exists(path)
    pd.DataFrame([row_dict]).to_csv(
        path,
        mode="a",
        header=not exists,
        index=False,
        quoting=1  # QUOTE_ALL
    )

# ========== MAIN LOOP（带断点续跑） ==========

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=str, nargs="+", required=True, help="JSONL input files (support globs by shell)")

    ap.add_argument("--model_type", type=str, required=True,
                    choices=["openai", "deepseek"])

    ap.add_argument("--model_paths", type=str, nargs="+", required=True, help="Model paths")
    ap.add_argument("--api_key", type=str, default=os.environ.get("API_KEY", ""), help="API key")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_csv", type=str, required=True, help="trial-wise CSV (append-safe)")
    ap.add_argument("--out_agg", type=str, required=True, help="aggregate CSV (recomputed at end)")
    # 健壮性参数
    ap.add_argument("--ignore_proxy", action="store_true",
                    help="Ignore HTTP(S)_PROXY from environment")
    ap.add_argument("--max_retries", type=int, default=5,
                    help="Max retries for network/HTTP errors")
    ap.add_argument("--delay", type=float, default=0.4,
                    help="Sleep seconds between trials to avoid QPS spikes")
    args = ap.parse_args()

    random.seed(args.seed)

    # === 加载 checkpoint（已完成的 trial/AGG） ===
    done_trials = set()
    if os.path.exists(args.out_csv):
        try:
            done_df = pd.read_csv(args.out_csv, dtype={"trial": str})
            for _, r in done_df.iterrows():
                done_trials.add((r["dataset"], r["model"], r["qid"], str(r["trial"])))
            print(f"[Checkpoint] Loaded {len(done_trials)} finished rows from {args.out_csv}")
        except Exception as e:
            print(f"[Checkpoint] Failed to read existing {args.out_csv}: {e}. Resume may rewrite duplicates.")

    agg_records = []

    for model_path in args.model_paths:
        llm = mk_client(
            args.model_type, model_path, args.api_key, args.temperature,
            ignore_proxy=args.ignore_proxy, max_retries=args.max_retries
        )

        for infile in args.inputs:
            dataset = os.path.basename(infile).replace(".jsonl", "")
            items = list(iter_jsonl(infile))
            n_items = len(items)

            votes_all = {"A": 0, "B": 0, "C": 0, "D": 0}

            for rec in tqdm(items, desc=f"Testing {model_path} @ {dataset}"):
                qid = rec.get("qid")
                strategies = rec["comfort_strategies"]
                opts = ["A", "B", "C", "D"]
                random.shuffle(opts)
                mapping = {
                    "A": strategies["A_direct_action"],
                    "B": strategies["B_indirect_relation"],
                    "C": strategies["C_hierarchy_deferential"],
                    "D": strategies["D_passive_avoidant"],
                }

                item_votes = {"A": 0, "B": 0, "C": 0, "D": 0}

                # ===== trials =====
                for t in range(args.trials):
                    key = (dataset, model_path, qid, str(t+1))
                    if key in done_trials:
                        continue

                    prompt = (
                        f"Scenario:\n{rec['scenario']}\n\n"
                        f"Choose the most appropriate comfort response:\n"
                        + "\n".join([f"{o}: {mapping[o]}" for o in opts])
                        + "\n\nAnswer with only the letter (A/B/C/D)."
                    )
                    resp = llm.invoke([{"role": "user", "content": prompt}])
                    choice = resp.strip()[0].upper() if resp else "?"

                    row = {
                        "dataset": dataset,
                        "model": model_path,
                        "qid": qid,
                        "trial": t + 1,
                        "choice": choice,
                        "raw": resp
                    }
                    append_row_csv(args.out_csv, row)
                    done_trials.add(key)

                    if choice in item_votes:
                        item_votes[choice] += 1
                        votes_all[choice] += 1

                    if args.delay > 0:
                        time.sleep(args.delay)

                # ===== AGG 行 =====
                agg_key = (dataset, model_path, qid, "AGG")
                if agg_key not in done_trials:
                    total_item_votes = sum(item_votes.values())
                    pA, pB, pC, pD = (
                        item_votes[x] / total_item_votes if total_item_votes else 0
                        for x in ["A", "B", "C", "D"]
                    )
                    item_entropy = compute_entropy([pA, pB, pC, pD])
                    item_indiv_idx = compute_individualism_index(pA, pB, pC, pD)

                    row = {
                        "dataset": dataset,
                        "model": model_path,
                        "qid": qid,
                        "trial": "AGG",
                        "choice": max(item_votes, key=item_votes.get) if total_item_votes else "?",
                        "votes_A": item_votes["A"],
                        "votes_B": item_votes["B"],
                        "votes_C": item_votes["C"],
                        "votes_D": item_votes["D"],
                        "entropy_bits": item_entropy,
                        "individualism_index": item_indiv_idx,
                    }
                    append_row_csv(args.out_csv, row)
                    done_trials.add(agg_key)

            # ===== dataset 级聚合 =====
            total_votes = sum(votes_all.values())
            pA, pB, pC, pD = (votes_all[x] / total_votes if total_votes else 0 for x in ["A", "B", "C", "D"])
            entropy = compute_entropy([pA, pB, pC, pD])
            indiv_idx = compute_individualism_index(pA, pB, pC, pD)

            agg_records.append({
                "dataset": dataset,
                "model": model_path,
                "n_items": n_items,
                "share_A": pA,
                "share_B": pB,
                "share_C": pC,
                "share_D": pD,
                "entropy_bits": entropy,
                "individualism_index": indiv_idx,
            })

    pd.DataFrame(agg_records).to_csv(args.out_agg, index=False)
    print(f"> Trial-wise appended to: {args.out_csv}")
    print(f"> Wrote aggregate: {args.out_agg}")

if __name__ == "__main__":
    main()
