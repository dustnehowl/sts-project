from datasets import load_dataset
import json
import os

def normalize(score: float) -> float:
    return round(score / 5.0, 4)  # 0~5 → 0~1 정규화

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def prepare_klue_sts():
    dataset = load_dataset("klue", "sts")

    os.makedirs("data", exist_ok=True)

    for split in ['train', 'validation']:
        processed = []
        for ex in dataset[split]:
            label = ex['labels']
            if label['label'] == -1:
                continue  # test set에는 label이 없음
            processed.append({
                'sentence1': ex['sentence1'],
                'sentence2': ex['sentence2'],
                'score': normalize(label['label'])  # 0~1로 정규화
            })

        save_jsonl(processed, f"data/{split}.json")
        print(f"Saved {split} set: {len(processed)} examples")


if __name__ == "__main__":
    prepare_klue_sts()
