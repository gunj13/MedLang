import json


# Step 2: Merge JSONL files, adding only missing questions
preg_jsonl = "preg_dataset_clean.jsonl"
mother_jsonl = "mother_question_and_answer_pairs_data.jsonl"
output_jsonl = "merged_preg_dataset.jsonl"

# Load existing questions from preg_dataset_clean.jsonl
preg_questions = set()
with open(preg_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        preg_questions.add(obj["question"].strip())

# Write all original preg_dataset_clean.jsonl entries to output
with open(output_jsonl, "w", encoding="utf-8") as out_f:
    with open(preg_jsonl, "r", encoding="utf-8") as in_f:
        for line in in_f:
            out_f.write(line)

    # Add only missing questions from mother_question_and_answer_pairs_data.jsonl
    with open(mother_jsonl, "r", encoding="utf-8") as mother_f:
        for line in mother_f:
            obj = json.loads(line)
            q = obj["question"].strip()
            if q not in preg_questions:
                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Merged dataset written to {output_jsonl}")


