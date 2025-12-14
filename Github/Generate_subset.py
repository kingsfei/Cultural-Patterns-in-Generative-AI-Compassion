import os
import json
from pathlib import Path
import argparse
from typing import Dict, Any, Iterable, List
from time import sleep

from dotenv import load_dotenv
from tqdm.auto import tqdm
import pandas as pd
from langchain_openai import ChatOpenAI

load_dotenv("./.env")

# ---------------- Prompts ----------------

SYSTEM_GEN = """You are a culturally-aware support strategist.

Design four distinct comfort strategies. Each strategy must be a single actionable
utterance or step (≤ 40 words) and ethically safe.

The strategies are grounded in the following behavioral definitions:

A: Direct-Action (individualistic, low-context, low power distance).
Explicitly identify the problem and offer immediate, concrete assistance using
first-person agency (e.g., “I can …”, “Want me to …?”).

B: Indirect-Relation (collectivistic, high-context).
Emphasize relational presence and face preservation through warmth and shared,
low-pressure rituals (e.g., sitting together, tea, a short walk), rather than
direct problem-solving.

C: Hierarchy-Deferential (high power distance, collectivistic).
Route support through formal roles, procedures, or authority structures
(e.g., requesting feedback or a formal review), without inventing roles not
explicitly stated in the scenario.

D: Passive-Avoidant (conflict-avoidant, collectivistic).
Reduce pressure by normalizing delay, space, or non-engagement
(e.g., “no rush”, “we can pause”), without dismissing feelings or forcing engagement.

General constraints:
- Do not contradict the scenario facts.
- Avoid medical or clinical claims.
- Use the same language as the scenario.
- Return strictly valid JSON only.
"""

USER_GEN = """Given the compassion-relevant item below, produce STRICT JSON:

{
  "comfort_strategies": {
    "A_direct_action": string,
    "B_indirect_relation": string,
    "C_hierarchy_deferential": string,
    "D_passive_avoidant": string
  }
}

Item:
{item_json}
"""

SYSTEM_POLISH = """You improve comfort strategies for compassion support.

Rules:
- Keep the JSON schema and keys exactly.
- Respond in the same language as the scenario.
- Each strategy must be ≤ 40 words.

Behavioral alignment:
- A: explicit, concrete help or action.
- B: relational presence in a high-context, face-preserving tone.
- C: formal channel, authority, or procedure; do not invent roles not stated.
- D: allow space, delay, or non-engagement without pressure.

Avoid repetition across B and D. Vary relational rituals when applicable.
"""

USER_POLISH = """Polish the comfort_strategies for the given item.

Return STRICT JSON:
{
  "comfort_strategies": {
    "A_direct_action": string,
    "B_indirect_relation": string,
    "C_hierarchy_deferential": string,
    "D_passive_avoidant": string
  }
}

Item:
{item_json}

Current:
{current_json}
"""

AUTHORITY_MARKERS = [
    "boss","manager","supervisor","teacher","professor","advisor","parent","parents",
    "dad","mom","principal","counselor","hr","committee","jury","coach"
]

# ---------------- LLM utils ----------------

def mk_client(model_type: str, model_path: str, temperature: float = 0.2) -> ChatOpenAI:
    if model_type == "openai-compatible":
        return ChatOpenAI(
            model=model_path,
            base_url=os.environ.get("API_URL"),
            api_key=os.environ.get("API_KEY"),
            temperature=temperature,
        )
    else:
        return ChatOpenAI(
            model=model_path,
            api_key=os.environ.get("API_KEY"),
            temperature=temperature,
        )

def safe_json(s: str) -> Dict[str, Any]:
    s = s.strip()
    if "```" in s:
        beg = s.find("{")
        end = s.rfind("}")
        s = s[beg:end + 1]
    return json.loads(s)

# ---------------- Generation ----------------

def call_gen(llm: ChatOpenAI, rec: Dict[str, Any], retries: int = 5) -> Dict[str, Any]:
    item = {
        "qid": rec.get("qid"),
        "scenario": rec.get("scenario", ""),
        "subject": rec.get("subject", ""),
        "emotion_label": rec.get("emotion_label", rec.get("label", "")),
    }
    u = USER_GEN.format(item_json=json.dumps(item, ensure_ascii=False))

    for i in range(retries):
        try:
            resp = llm.invoke([
                {"role": "system", "content": SYSTEM_GEN},
                {"role": "user", "content": u},
            ])
            data = safe_json(resp.content if hasattr(resp, "content") else str(resp))
            cs = data.get("comfort_strategies", {})
            for k in ["A_direct_action","B_indirect_relation",
                      "C_hierarchy_deferential","D_passive_avoidant"]:
                assert isinstance(cs.get(k, ""), str) and cs.get(k, "").strip()
            return cs
        except Exception:
            if i < retries - 1:
                sleep(3)
            else:
                raise

def call_polish(llm: ChatOpenAI, rec: Dict[str, Any],
                strategies: Dict[str, str], retries: int = 2) -> Dict[str, Any]:
    item = {
        "qid": rec.get("qid"),
        "scenario": rec.get("scenario", ""),
        "subject": rec.get("subject", ""),
        "emotion_label": rec.get("emotion_label", rec.get("label", "")),
    }
    u = USER_POLISH.format(
        item_json=json.dumps(item, ensure_ascii=False),
        current_json=json.dumps({"comfort_strategies": strategies}, ensure_ascii=False)
    )

    for i in range(retries):
        try:
            resp = llm.invoke([
                {"role": "system", "content": SYSTEM_POLISH},
                {"role": "user", "content": u},
            ])
            data = safe_json(resp.content if hasattr(resp, "content") else str(resp))
            cs = data.get("comfort_strategies", {})
            for k in ["A_direct_action","B_indirect_relation",
                      "C_hierarchy_deferential","D_passive_avoidant"]:
                assert isinstance(cs.get(k, ""), str) and cs.get(k, "").strip()
            return cs
        except Exception:
            if i < retries - 1:
                sleep(2)
            else:
                return strategies

# ---------------- IO helpers ----------------

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", type=str, required=True)
    ap.add_argument("--output_jsonl", type=str, required=True)
    ap.add_argument("--output_csv", type=str, required=True)
    ap.add_argument("--model_type", type=str, default="openai",
                    choices=["openai","openai-compatible"])
    ap.add_argument("--model_path", type=str, default="gpt-4o-mini")
    ap.add_argument("--num_retries", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    llm = mk_client(args.model_type, args.model_path, temperature=0.1)

    out_rows: List[Dict[str, Any]] = []
    processed = 0

    for rec in tqdm(iter_jsonl(args.input_path), desc="Generating ABCD strategies"):
        processed += 1
        cs = call_gen(llm, rec, retries=args.num_retries)
        cs = call_polish(llm, rec, cs, retries=2)

        out_rec = dict(rec)
        out_rec["comfort_strategies"] = cs
        out_rows.append(out_rec)

        if args.limit and processed >= args.limit:
            break

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rows = []
    for r in out_rows:
        cs = r.get("comfort_strategies", {})
        rows.append({
            "qid": r.get("qid"),
            "scenario": r.get("scenario",""),
            "subject": r.get("subject",""),
            "emotion_label": r.get("emotion_label", r.get("label","")),
            "comfort_A_direct_action": cs.get("A_direct_action",""),
            "comfort_B_indirect_relation": cs.get("B_indirect_relation",""),
            "comfort_C_hierarchy_deferential": cs.get("C_hierarchy_deferential",""),
            "comfort_D_passive_avoidant": cs.get("D_passive_avoidant",""),
        })
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    print(f"> Processed: {processed}")
    print(f"> Wrote JSONL: {args.output_jsonl}")
    print(f"> Wrote CSV  : {args.output_csv}")

if __name__ == "__main__":
    main()
