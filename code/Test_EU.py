# Qwen_Test_EU.py
import os
import json
import argparse
import pandas as pd
from tqdm.auto import tqdm

from utils import load_yaml, save_gen_results
from data import DataLoader
from model import LLM

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def rank_choices(choices):
    choices = choices or []
    return "\n".join([f"{LETTERS[i]}) {c}" for i, c in enumerate(choices)])


def load_dataset(path: str, lang: str) -> pd.DataFrame:
    df = pd.read_json(path_or_buf=path, lines=True, encoding="utf-8")
    if lang:
        df = df[df.get("language") == lang]
    return df


def _safe_col(df: pd.DataFrame, col: str):
    return df[col] if col in df.columns else pd.Series([None] * len(df))


def filter_checkpoint(task: str, model_name: str, lang: str, df: pd.DataFrame) -> pd.DataFrame:

    file_path = f"results/{task}/{model_name}.jsonl"
    if not os.path.exists(file_path):
        return df
    try:
        prev = pd.read_json(path_or_buf=file_path, lines=True, encoding="utf-8")
        prev = prev[prev.get("lang") == lang]

        if task == "EA":
            completed = prev[_safe_col(prev, "answer").astype(str).str.len() > 0]
        else:  # EU
            emo_ok = _safe_col(prev, "emo_answer").astype(str).str.len() > 0
            cause_ok = _safe_col(prev, "cause_answer").astype(str).str.len() > 0
            completed = prev[emo_ok & cause_ok]

        done_ids = set(completed.get("qid", []))
        print(f"> Found checkpoint with {len(completed)} completed samples; "
              f"will retry {max(0, len(prev) - len(completed))} incomplete ones")
        return df[~df.get("qid").isin(done_ids)].copy()
    except Exception:
        return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to EU/EA jsonl, e.g., data/EU_German.jsonl")
    parser.add_argument("--dataset_lang", type=str, required=True, help="Language code in dataset, e.g., de/ja/en/zh")
    parser.add_argument("--prompt_lang", type=str, default="en", choices=["en", "zh", "de", "ja"], help="Prompt language for system/instruction")
    parser.add_argument("--schema", type=str, default="EU", choices=["EU", "EA"], help="Dataset schema: EU or EA")
    parser.add_argument("--token", type=str, default=os.environ.get("SILICONFLOW_TOKEN", ""), help="SiliconFlow token (or set SILICONFLOW_TOKEN)")
    parser.add_argument("--model", type=str, default="Qwen/QwQ-480B", help="Qwen model name on SiliconFlow, e.g., Qwen/QwQ-480B")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--use_cot", action="store_true", default=False)
    parser.add_argument("--num_retries", type=int, default=5)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--start_qid", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    if not args.token:
        print("Missing token. Provide --token or set SILICONFLOW_TOKEN.")
        return
    os.environ["OPENAI_API_KEY"] = args.token
    os.environ["OPENAI_BASE_URL"] = SILICONFLOW_BASE_URL

    llm = LLM(
        model_type="openai-compatible",
        model_path=args.model,
        num_retries=args.num_retries,
        device=args.device,
        use_cot=args.use_cot,
        eval_only=args.eval_only,
        debug=args.debug,
    )

    llm.model_name = f"{args.model.replace('/', '_')}-{args.dataset_lang}"

    task = args.schema

    prompts = load_yaml("src/configs/prompts.yaml")
    prompt_tpl = prompts[task][args.prompt_lang]

    if not args.eval_only:
        llm.init_prompt(task, args.prompt_lang)
        df = load_dataset(args.dataset_path, args.dataset_lang)
        if args.start_qid:
            df = df[df["qid"].astype(str) >= str(args.start_qid)]
        df = filter_checkpoint(task, llm.model_name, args.dataset_lang, df)
        if args.limit:
            df = df.head(args.limit)

        print(f"> Loaded custom {task} dataset ({args.dataset_lang}) with {len(df)} samples")
        print(f"> Generating {task}-{args.dataset_lang} with prompt lang {args.prompt_lang} for {llm.model_name}...")

        # debug 文件
        os.makedirs(f"results/{task}", exist_ok=True)
        debug_file = f"results/{task}/{llm.model_name}.debug.jsonl"

        for _, sample in tqdm(df.iterrows(), total=len(df)):
            if task == "EU":
                # --- 构造 EU 提示 ---
                msg = prompt_tpl.format(
                    scenario=sample["scenario"],
                    subject=sample["subject"],
                    emo_choices=rank_choices(sample["emotion_choices"]),
                    cause_choices=rank_choices(sample["cause_choices"]),
                )
                # 兜底模板（按提示语言）
                if not str(msg).strip():
                    if args.prompt_lang == "ja":
                        msg = (
                            f"## シナリオ\n{sample['scenario']}\n\n"
                            f"## 質問 1\n{sample['subject']} はどの感情？\n\n"
                            f"## 選択肢1\n{rank_choices(sample['emotion_choices'])}\n\n"
                            f"## 質問 2\n理由は？\n\n"
                            f"## 選択肢2\n{rank_choices(sample['cause_choices'])}"
                        )
                    elif args.prompt_lang == "zh":
                        msg = (
                            f"## 场景\n{sample['scenario']}\n\n"
                            f"## 问题 1\n{sample['subject']} 最终会感受到什么情绪？\n\n"
                            f"## 选项1\n{rank_choices(sample['emotion_choices'])}\n\n"
                            f"## 问题 2\n为什么？\n\n"
                            f"## 选项2\n{rank_choices(sample['cause_choices'])}"
                        )
                    else:  # en/de 用英文兜底
                        msg = (
                            f"## Scenario\n{sample['scenario']}\n\n"
                            f"## Question 1\nWhat emotion(s) would {sample['subject']} feel?\n\n"
                            f"## Choices Q1\n{rank_choices(sample['emotion_choices'])}\n\n"
                            f"## Question 2\nWhy?\n\n"
                            f"## Choices Q2\n{rank_choices(sample['cause_choices'])}"
                        )

                response = llm.gen_response(msg)

                if args.debug:
                    with open(debug_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(
                            {"qid": sample["qid"], "prompt": msg, "response": response},
                            ensure_ascii=False
                        ) + "\n")

                # label -> 字母
                try:
                    emo_label_letter = LETTERS[sample["emotion_choices"].index(sample["emotion_label"])]
                except Exception:
                    emo_label_letter = ""
                try:
                    cause_label_letter = LETTERS[sample["cause_choices"].index(sample["cause_label"])]
                except Exception:
                    cause_label_letter = ""

                res = {
                    "qid": sample["qid"],
                    "lang": sample.get("language", ""),
                    "coarse_category": sample.get("coarse_category", ""),
                    "finegrained_category": sample.get("finegrained_category", ""),
                    "emo_label": emo_label_letter,
                    "emo_answer": (response or {}).get("answer_q1", ""),
                    "cause_label": cause_label_letter,
                    "cause_answer": (response or {}).get("answer_q2", ""),
                }

            else:  # EA
                msg = prompt_tpl.format(
                    scenario=sample["scenario"],
                    subject=sample["subject"],
                    choices=rank_choices(sample["choices"]),
                    q_type=sample.get("question type", sample.get("question_type", "")),
                )
                if not str(msg).strip():
                    q_type = sample.get("question type", sample.get("question_type", "strategy"))
                    msg = (
                        f"## Scenario\n{sample['scenario']}\n\n"
                        f"## Question\nWhat is the most effective {q_type} for {sample['subject']}?\n\n"
                        f"## Choices\n{rank_choices(sample['choices'])}"
                    )

                response = llm.gen_response(msg)

                if args.debug:
                    with open(debug_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(
                            {"qid": sample["qid"], "prompt": msg, "response": response},
                            ensure_ascii=False
                        ) + "\n")

                label_letter = ""
                if sample.get("label") in (sample.get("choices") or []):
                    try:
                        label_letter = LETTERS[sample["choices"].index(sample["label"])]
                    except Exception:
                        label_letter = ""

                res = {
                    "qid": sample["qid"],
                    "lang": sample.get("language", ""),
                    "category": sample.get("category", ""),
                    "label": label_letter,
                    "answer": (response or {}).get("answer", ""),
                }

            save_gen_results(res, task, llm.model_name)

    # --- 评测阶段（沿用你现有的 DataLoader 逻辑）---
    dl = DataLoader(model=llm, task=task, lang=args.dataset_lang, eval_only=True)
    dl.evaluate_results()


if __name__ == "__main__":
    main()
