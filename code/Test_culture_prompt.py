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

# ==================== PRIMES ====================
PRIMES = {
    "chinese": {
        "en": "He/She is an average person born and living in the China. For the following scenario, choose the most appropriate action for him/her.",
        "de": "Er/Sie ist eine Person, die in China geboren wurde und dort lebt. Wählen Sie für das folgende Szenario die am besten geeignete Aktion für ihn/sie aus.",
        "ja": "彼/彼女は中国で生まれ、中国で生活している人です。次のシナリオについて、彼/彼女にとって最も適切な行動を選んでください。",
        "zh": "他/她是在中国出生并在中国生活的人。请在下列情境中为他/她选择最合适的行动。"
    },
    "american": {
        "en": "He/She is an average person born and living in the United States. For the following scenario, choose the most appropriate action for him/her.",
        "de": "Er/Sie ist eine Person, die in den Vereinigten Staaten geboren wurde und dort lebt. Wählen Sie für das folgende Szenario die am besten geeignete Aktion für ihn/sie aus.",
        "ja": "彼/彼女はアメリカ合衆国で生まれ、そこで生活している人です。次のシナリオについて、彼/彼女にとって最も適切な行動を選んでください。",
        "zh": "他/她是在美国出生并长期生活在美国的人。请在下列情境中为他/她选择最恰当的行动。"
    },
    "none": {"en":"", "de":"", "ja":"", "zh":""}
}

def make_prime_text(prime_type, lang):
    prime_type = (prime_type or "none").lower()
    lang = (lang or "en").lower()
    if prime_type not in PRIMES:
        prime_type = "none"
    if lang not in PRIMES[prime_type]:
        lang = "en"
    return PRIMES[prime_type][lang].strip()

def infer_language_from_dataset_name(name):
    s = (name or "").lower()
    for key in ["en","de","ja","zh"]:
        if f"_{key}_" in s or s.endswith(f"_{key}"):
            return key
    return "en"

def native_language_for_prime(prime_type):
    pt = (prime_type or "").lower()
    return {"chinese": "zh", "american": "en"}.get(pt, "")

def build_auto_inputs(auto_dir, prime_type):
    lang_native = native_language_for_prime(prime_type)
    all_langs = ["en","de","ja","zh"]
    langs = [l for l in all_langs if l != lang_native] if lang_native else all_langs
    files = []
    for l in langs:
        path = os.path.join(auto_dir, f"compassion_subset_{l}_with_strat.jsonl")
        if os.path.exists(path):
            files.append(path)
    if not files:
        raise FileNotFoundError(f"No input JSONL found in {auto_dir}")
    print(f"[AUTO] prime={prime_type}, skip native={lang_native or 'none'}, inputs={files}")
    return files

def mk_client(
    model_type: str,
    model_path: str,
    api_key: str,
    temperature: float = 1.0,
    timeout: int = 60,
    max_retries: int = 5,
    delay: float = 0.6,
    ignore_proxy: bool = False
):

    if (model_type or "").lower() != "siliconflow":
        raise ValueError("This build only supports model_type='siliconflow'")

    def _mk_session(ignore_proxy: bool, max_retries: int):
        sess = requests.Session()
        if ignore_proxy:
            sess.trust_env = False
            sess.proxies = {"http": None, "https": None}
        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["POST", "GET"]),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        return sess

    url = "https://api.siliconflow.cn/v1/chat/completions"
    sess = _mk_session(ignore_proxy, max_retries)

    class Wrapper:
        def invoke(self, messages):
            payload = {
                "model": model_path,
                "messages": messages,
                "temperature": temperature,
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "CompassionBench/1.2",
            }
            attempts = 0
            last_err = None
            while True:
                attempts += 1
                try:
                    r = sess.post(url, headers=headers, json=payload, timeout=timeout)
                    if r.status_code == 401:
                        raise SystemExit("[FATAL] 401 Unauthorized from SiliconFlow. Stopping to preserve progress.")
                    if r.status_code in (429, 500, 502, 503, 504):
                        wait = min(8.0, delay * (2 ** (attempts - 1))) + random.uniform(0, 0.8)
                        print(f"[WARN] HTTP {r.status_code}: {r.text[:180]} | retry {attempts}/{max_retries} after {wait:.1f}s")
                        time.sleep(wait)
                        last_err = RuntimeError(f"HTTP {r.status_code}")
                        if attempts < max_retries:
                            continue
                    r.raise_for_status()
                    data = r.json()
                    return data["choices"][0]["message"]["content"]
                except SystemExit:
                    raise
                except requests.exceptions.RequestException as e:
                    last_err = e
                    if attempts >= max_retries:
                        raise
                    wait = min(8.0, delay * (2 ** (attempts - 1))) + random.uniform(0, 0.8)
                    print(f"[WARN] Request failed ({type(e).__name__}): {e} | retry {attempts}/{max_retries} after {wait:.1f}s")
                    time.sleep(wait)

    return Wrapper()

# ==================== HELPERS ====================
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
    exists = os.path.exists(path)
    pd.DataFrame([row_dict]).to_csv(path, mode="a", header=not exists, index=False)

def parse_choice(text: str) -> str:

    import re
    if not text:
        return "?"
    up = text.upper()
    m = re.search(r'\b([ABCD])\b', up)
    if m:
        return m.group(1)
    m = re.search(r'^\s*[\(\[]?\s*([ABCD])\s*[\)\]]?', up)
    if m:
        return m.group(1)
    return "?"

# ==================== MAIN ====================
def main():
    ap = argparse.ArgumentParser()
    # 输入控制：支持自动选择 or 手工传入
    ap.add_argument("--inputs", type=str, nargs="+", default=[], help="Explicit JSONL input files (supports shell globs)")
    ap.add_argument("--auto_inputs_dir", type=str, default="", help="If set, auto-pick compassion_subset_{lang}_with_strat.jsonl (skip native of prime)")
    # 模型/调用控制
    ap.add_argument("--model_type", type=str, required=True, choices=["siliconflow"])
    ap.add_argument("--model_paths", type=str, nargs="+", required=True, help="Model paths for SiliconFlow (e.g., deepseek-ai/DeepSeek-R1)")
    ap.add_argument("--api_key", type=str, default=os.environ.get("API_KEY", ""), help="SiliconFlow API key")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--delay", type=float, default=0.6, help="Sleep seconds between trials to avoid QPS spikes")
    ap.add_argument("--ignore_proxy", action="store_true")
    # prime
    ap.add_argument("--prime_type", type=str, default="none", choices=["none","chinese","american"])
    # 输出
    ap.add_argument("--out_csv", type=str, required=True, help="trial-wise CSV (append-safe; also includes per-item AGG rows)")
    ap.add_argument("--out_agg", type=str, required=True, help="aggregate CSV (per (dataset,model))")
    # 可选：将“只输出字母”放 system（更稳）
    ap.add_argument("--use_system_guard", action="store_true", help="Put strict instruction in system role for better compliance")
    args = ap.parse_args()

    # 代理处理
    if args.ignore_proxy:
        for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","all_proxy"]:
            os.environ.pop(k, None)

    # 自动选择输入文件
    if args.auto_inputs_dir:
        input_files = build_auto_inputs(args.auto_inputs_dir, args.prime_type)
    else:
        input_files = args.inputs
    if not input_files:
        raise FileNotFoundError("No input files. Provide --inputs or --auto_inputs_dir.")

    # 断点续跑（读取已完成 trial/AGG）
    done_records = set()
    if os.path.exists(args.out_csv):
        try:
            df_done = pd.read_csv(args.out_csv, dtype={"trial": str})
            done_records = set((row["model"], row["dataset"], str(row["qid"]), str(row["trial"]))
                               for _, row in df_done.iterrows())
            print(f"[RESUME] Loaded {len(done_records)} completed rows from {args.out_csv}")
        except Exception as e:
            print(f"[WARN] Could not read previous results: {e}")

    # 初始化输出目录
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_agg), exist_ok=True)
    if not os.path.exists(args.out_csv):
        pd.DataFrame(columns=[
            "dataset","language","model","qid","trial","choice","raw",
            "prime_type","prime_text","votes_A","votes_B","votes_C","votes_D",
            "entropy_bits","individualism_index"
        ]).to_csv(args.out_csv, index=False)

    agg_records = []

    # 主循环
    for model_path in args.model_paths:
        llm = mk_client(
            args.model_type, model_path, args.api_key, args.temperature,
            timeout=args.timeout, max_retries=args.max_retries, delay=args.delay,
            ignore_proxy=args.ignore_proxy
        )

        for infile in input_files:
            dataset = os.path.basename(infile).replace(".jsonl","")
            lang = infer_language_from_dataset_name(dataset)
            prime_text = make_prime_text(args.prime_type, lang)
            items = list(iter_jsonl(infile))
            votes_all = {"A":0,"B":0,"C":0,"D":0}

            for rec in tqdm(items, desc=f"{model_path} @ {dataset}"):
                qid = str(rec.get("qid"))
                strategies = rec["comfort_strategies"]
                mapping = {
                    "A": strategies["A_direct_action"],
                    "B": strategies["B_indirect_relation"],
                    "C": strategies["C_hierarchy_deferential"],
                    "D": strategies["D_passive_avoidant"],
                }
                opts = ["A","B","C","D"]
                random.shuffle(opts)
                item_votes = {"A":0,"B":0,"C":0,"D":0}

                # 多次抽样
                for t in range(args.trials):
                    key = (model_path, dataset, qid, str(t+1))
                    if key in done_records:
                        continue

                    # 组装 messages（可选把 guard 放 system）
                    user_prompt = (
                        (prime_text + "\n\n") if prime_text else ""
                    ) + (
                        f"Scenario:\n{rec['scenario']}\n\n"
                        f"Choose the most appropriate comfort response:\n"
                        + "\n".join([f"{o}: {mapping[o]}" for o in opts])
                        + "\n\nAnswer with only the letter (A/B/C/D)."
                    )
                    messages = []
                    if args.use_system_guard:
                        messages.append({"role":"system","content":
                            "You are a careful evaluator. Read the scenario and four options. "
                            "Return ONLY one letter among A/B/C/D. No explanations."})
                        if prime_text:
                            # prime 也可放 system，不冲突
                            messages.append({"role":"system","content": prime_text})
                        # user 只放题目与选项
                        messages.append({"role":"user","content": (
                            f"Scenario:\n{rec['scenario']}\n\n"
                            f"Choose the most appropriate comfort response:\n"
                            + "\n".join([f"{o}: {mapping[o]}" for o in opts])
                            + "\n\nAnswer with only the letter (A/B/C/D)."
                        )})
                    else:
                        # 将 prime + 题目一起放 user（保持与你原始逻辑一致）
                        messages.append({"role":"user","content": user_prompt})

                    try:
                        resp = llm.invoke(messages)
                    except SystemExit as e:
                        print(str(e))
                        print("[STOP] Progress saved. Rerun later to resume.")
                        return
                    except Exception as e:
                        resp = str(e)

                    choice = parse_choice(resp)

                    # 写 trial 行
                    append_row_csv(args.out_csv, {
                        "dataset": dataset, "language": lang, "model": model_path,
                        "qid": qid, "trial": t+1, "choice": choice, "raw": resp,
                        "prime_type": args.prime_type, "prime_text": prime_text
                    })
                    done_records.add(key)

                    if choice in item_votes:
                        item_votes[choice] += 1
                        votes_all[choice] += 1

                    if args.delay > 0:
                        time.sleep(args.delay)

                # 每题的 AGG 行
                agg_key = (model_path, dataset, qid, "AGG")
                if agg_key not in done_records:
                    total_item_votes = sum(item_votes.values())
                    if total_item_votes:
                        pA,pB,pC,pD = [item_votes[x]/total_item_votes for x in "ABCD"]
                        item_entropy = compute_entropy([pA,pB,pC,pD])
                        item_indiv = compute_individualism_index(pA,pB,pC,pD)
                        maj_choice = max(item_votes, key=item_votes.get)
                    else:
                        pA=pB=pC=pD=item_entropy=item_indiv=0
                        maj_choice = "?"

                    append_row_csv(args.out_csv, {
                        "dataset": dataset, "language": lang, "model": model_path,
                        "qid": qid, "trial": "AGG", "choice": maj_choice,
                        "votes_A": item_votes["A"], "votes_B": item_votes["B"],
                        "votes_C": item_votes["C"], "votes_D": item_votes["D"],
                        "entropy_bits": item_entropy, "individualism_index": item_indiv,
                        "prime_type": args.prime_type, "prime_text": prime_text
                    })
                    done_records.add(agg_key)

            total_votes = sum(votes_all.values())
            if total_votes:
                pA,pB,pC,pD = [votes_all[x]/total_votes for x in "ABCD"]
                entropy = compute_entropy([pA,pB,pC,pD])
                indiv = compute_individualism_index(pA,pB,pC,pD)
            else:
                pA=pB=pC=pD=entropy=indiv=0

            agg_records.append({
                "dataset": dataset, "language": lang, "model": model_path,
                "share_A": pA, "share_B": pB, "share_C": pC, "share_D": pD,
                "entropy_bits": entropy, "individualism_index": indiv,
                "prime_type": args.prime_type
            })

    pd.DataFrame(agg_records).to_csv(args.out_agg, index=False)
    print(f"> Trial-wise & per-item AGG appended to: {args.out_csv}")
    print(f"> Wrote aggregate: {args.out_agg}")

if __name__ == "__main__":
    main()
