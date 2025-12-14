import os
import json
import argparse
from typing import List, Dict, Any
from time import sleep

from dotenv import load_dotenv
from tqdm.auto import tqdm
from langchain_openai import ChatOpenAI


load_dotenv("./.env")


# =========================
# System Prompt
# =========================

def build_system_prompt(target_lang: str) -> str:
    return (
        "You are a professional translation assistant for multilingual dataset construction. "
        "Translate the provided fields into the target language while preserving semantic content, "
        "pragmatic intent, and the original level of formality, allowing only minimal surface-level "
        "linguistic adaptation when necessary for naturalness. "
        "Do NOT introduce new information, reinterpret the scenario, or modify the underlying intent. "
        "Do NOT perform reasoning, explanation, or cultural analysis. "
        "Return ONLY a valid JSON object that strictly matches the requested schema. "
        f"Target language: {target_lang}."
    )


# =========================
# User Prompt
# =========================

def build_user_prompt(record: Dict[str, Any], target_lang: str) -> str:
    payload = {
        "scenario": record.get("scenario", ""),
        "subject": record.get("subject", ""),
        "question_type": record.get("question type", ""),
        "choices": record.get("choices", []),
    }

    return (
        "Translate the following dataset fields into the target language.\n\n"
        "Constraints:\n"
        "- Preserve the original semantic meaning and pragmatic intent.\n"
        "- Maintain the original communicative function and level of formality.\n"
        "- You may apply minimal surface-level linguistic adaptation only when required for natural "
        "expression in the target language.\n"
        "- Do NOT add, remove, or reinterpret information.\n"
        "- Do NOT introduce culturally specific reasoning beyond what is explicitly stated.\n"
        "- Keep list lengths and item order identical.\n"
        "- Do NOT translate named entities unless strictly necessary.\n\n"
        "Return strictly the following JSON schema:\n\n"
        "```json\n"
        "{\n"
        "  \"scenario\": string,\n"
        "  \"subject\": string,\n"
        "  \"question_type\": string,\n"
        "  \"choices\": string[]\n"
        "}\n"
        "```\n\n"
        "Input JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


# =========================
# Translation Call
# =========================

def translate_fields(
    client: ChatOpenAI,
    record: Dict[str, Any],
    target_lang: str,
    retries: int
) -> Dict[str, Any]:

    sys_prompt = build_system_prompt(target_lang)
    user_prompt = build_user_prompt(record, target_lang)

    for attempt in range(retries):
        try:
            response = client.invoke([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ])

            content = response.content if hasattr(response, "content") else str(response)

            if "```" in content:
                start = content.find("{")
                end = content.rfind("}")
                content = content[start:end + 1]

            data = json.loads(content)

            if not isinstance(data.get("choices", None), list):
                raise ValueError("Invalid output: 'choices' is not a list")

            return data

        except Exception as e:
            if attempt < retries - 1:
                sleep(3)
            else:
                raise RuntimeError(f"Translation failed after {retries} attempts") from e


# =========================
# Label Mapping
# =========================

def map_label_by_index(
    original_choices: List[str],
    translated_choices: List[str],
    original_label: str
) -> str:
    try:
        idx = original_choices.index(original_label)
        return translated_choices[idx]
    except Exception:
        return original_label


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/EA.jsonl")
    parser.add_argument("--output_path", type=str, default="data/EA_translated.jsonl")
    parser.add_argument("--target_lang", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="openai",
                        choices=["openai", "openai-compatible"])
    parser.add_argument("--model_path", type=str, default="gpt-4o")
    parser.add_argument("--num_retries", type=int, default=5)
    parser.add_argument("--start_qid", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.model_type == "openai-compatible":
        client = ChatOpenAI(
            model=args.model_path,
            base_url=os.environ.get("API_URL"),
            api_key=os.environ.get("API_KEY"),
            temperature=0.2,
        )
    else:
        client = ChatOpenAI(
            model=args.model_path,
            api_key=os.environ.get("API_KEY"),
            temperature=0.2,
        )

    processed = 0
    started = args.start_qid is None

    with open(args.input_path, "r", encoding="utf-8") as fin, \
         open(args.output_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Translating EA"):
            if not line.strip():
                continue

            rec = json.loads(line)

            if not started:
                if str(rec.get("qid")) == str(args.start_qid):
                    started = True
                else:
                    continue

            original_choices = rec.get("choices", [])
            original_label = rec.get("label", "")

            translated = translate_fields(
                client=client,
                record=rec,
                target_lang=args.target_lang,
                retries=args.num_retries,
            )

            rec_out = {
                "qid": rec.get("qid"),
                "language": args.target_lang,
                "category": rec.get("category"),
                "question type": translated.get(
                    "question_type", rec.get("question type")
                ),
                "scenario": translated.get("scenario", rec.get("scenario")),
                "subject": translated.get("subject", rec.get("subject")),
                "choices": translated.get("choices", original_choices),
            }

            rec_out["label"] = map_label_by_index(
                original_choices=original_choices,
                translated_choices=rec_out["choices"],
                original_label=original_label,
            )

            fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

            processed += 1
            if args.limit and processed >= args.limit:
                break

    print(f"> Done. Wrote {processed} records to {args.output_path}")


if __name__ == "__main__":
    main()
