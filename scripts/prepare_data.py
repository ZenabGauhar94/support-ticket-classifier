from datasets import load_dataset
import json
import os

dataset = load_dataset("banking77")
label_names = dataset["train"].features["label"].names

def format_example(example):
    ticket_text = example["text"]
    label = label_names[example["label"]]
    prompt = (
        "Classify this customer support ticket into exactly one category.\n\n"
        f"Ticket: {ticket_text}\n\n"
        "Category:"
    )
    return {"prompt": prompt, "completion": f" {label}"}

os.makedirs("data", exist_ok=True)

for split in ["train", "test"]:
    formatted = [format_example(ex) for ex in dataset[split]]
    out_path = f"data/{split}.jsonl"
    with open(out_path, "w") as f:
        for row in formatted:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(formatted)} examples to {out_path}")


with open("data/labels.json", "w") as f:
    json.dump(label_names, f)