#!/usr/bin/env python3
"""
Конвертирует датасет Кристины (dataset_entity в формате conll с # text = и # relations =)
в JSON предложений для оценки (формат sentences.json: id, text, terms, relations).
Запуск из корня проекта: poetry run python scripts/convert_dataset_kristina_to_sentences.py [путь_к_dataset_entity] [выходной.json]
"""

import json
import os
import re
import sys
from pathlib import Path

# корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Маппинг классов датасета на онтологию проекта
CLASS_MAP = {
    "Technology": "Application",
    "App_system": "Application",
    "Subject": "Object",
}

# Нормализация имени предиката (датасет может использовать IsAlternativeNameFor)
def normalize_predicate(predicate: str) -> str:
    if predicate == "IsAlternativeNameFor":
        return "isAlternativeNameFor"
    return predicate


def extract_class_from_label(label: str) -> str:
    """B-Method_solves_Task -> Method, B-Science -> Science."""
    if not label.startswith("B-") and not label.startswith("I-"):
        return None
    part = label[2:].strip()
    if "_" in part:
        part = part.split("_")[0]
    return CLASS_MAP.get(part, part)


def find_token_spans(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    """Возвращает [(start, end), ...] для каждого токена в text."""
    spans = []
    pos = 0
    for token in tokens:
        # ищем токен в тексте (пропуская уже пройденную часть)
        idx = text.find(token, pos)
        if idx < 0:
            # пробуем без учёта позиции
            idx = text.find(token)
        if idx < 0:
            spans.append((pos, pos + len(token)))
            pos = pos + len(token) + 1
        else:
            start, end = idx, idx + len(token)
            spans.append((start, end))
            pos = end
    return spans


def parse_entity_file(path: Path) -> list[dict]:
    """
    Парсит один файл dataset_entity_*.txt.
    Возвращает список предложений: [{"text", "terms": [{class, value, start_pos}], "relations": [...]}]
    """
    content = path.read_text(encoding="utf-8")
    sentences_out = []
    # блоки по пустым строкам и # text =
    blocks = re.split(r"\n\s*\n", content)
    global_sent_id = 0
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        text = ""
        relations_str = ""
        token_lines = []
        for line in lines:
            if line.startswith("# text ="):
                text = line.replace("# text =", "").strip()
            elif line.startswith("# relations ="):
                relations_str = line.replace("# relations =", "").strip().strip('"')
            elif line.strip() and not line.startswith("#"):
                token_lines.append(line.strip())
        if not text or not token_lines:
            continue
        tokens = []
        labels = []
        for line in token_lines:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                tokens.append(parts[0])
                labels.append(parts[1])
            else:
                tokens.append(parts[0])
                labels.append("O")
        spans = find_token_spans(text, tokens)
        if len(spans) != len(tokens):
            continue
        terms_list = []
        i = 0
        while i < len(labels):
            label = labels[i]
            cls = extract_class_from_label(label)
            if cls is None:
                i += 1
                continue
            start, end = spans[i]
            j = i + 1
            while j < len(labels) and labels[j].startswith("I-") and extract_class_from_label(labels[j]) == cls:
                end = spans[j][1]
                j += 1
            value = text[start:end]
            terms_list.append({"class": cls, "value": value, "start_pos": start})
            i = j
        terms_by_class = {}
        for t in terms_list:
            c = t["class"]
            terms_by_class.setdefault(c, []).append(t)
        relations_out = []
        if relations_str:
            for part in relations_str.split(","):
                part = part.strip()
                if not part:
                    continue
                toks = part.split()
                if len(toks) != 3:
                    continue
                rel_id, idx1_s, idx2_s = toks
                try:
                    idx1, idx2 = int(idx1_s), int(idx2_s)
                except ValueError:
                    continue
                parts_rel = rel_id.split("_")
                if len(parts_rel) < 3:
                    continue
                class1, predicate = parts_rel[0], parts_rel[1]
                class2 = "_".join(parts_rel[2:])
                predicate = normalize_predicate(predicate)
                class1 = CLASS_MAP.get(class1, class1)
                class2 = CLASS_MAP.get(class2, class2)
                if class1 not in terms_by_class or class2 not in terms_by_class:
                    continue
                list1, list2 = terms_by_class[class1], terms_by_class[class2]
                if idx1 >= len(list1) or idx2 >= len(list2):
                    continue
                t1, t2 = list1[idx1], list2[idx2]
                relations_out.append({
                    "term1": {"class": t1["class"], "value": t1["value"], "start_pos": t1["start_pos"]},
                    "predicate": predicate,
                    "term2": {"class": t2["class"], "value": t2["value"], "start_pos": t2["start_pos"]},
                })
        global_sent_id += 1
        sentences_out.append({
            "id": global_sent_id,
            "text": text,
            "terms": terms_list,
            "relations": relations_out,
        })
    return sentences_out


def main():
    base = PROJECT_ROOT / "Dataset_Kristina"
    if not base.is_dir():
        base = PROJECT_ROOT.parent / "Dataset_Kristina"
    dataset_dir = base / "5. Датасет" / "dataset_for_extraction" / "dataset_entity"
    out_path = PROJECT_ROOT / "tests" / "dataset_kristina_sentences.json"
    if len(sys.argv) >= 2:
        dataset_dir = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2])
    if not dataset_dir.is_dir():
        print(f"Папка не найдена: {dataset_dir}")
        print("Укажите путь к папке dataset_entity первым аргументом.")
        sys.exit(1)
    all_sentences = []
    sent_id = 0
    for f in sorted(dataset_dir.glob("dataset_entity_*.txt")):
        for sent in parse_entity_file(f):
            sent_id += 1
            sent["id"] = sent_id
            all_sentences.append(sent)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"sentences": all_sentences}, f, ensure_ascii=False, indent=2)
    print(f"Сохранено {len(all_sentences)} предложений в {out_path}")


if __name__ == "__main__":
    main()
