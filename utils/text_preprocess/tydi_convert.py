
import json
import os
from tqdm import tqdm
import argparse
import jsonlines
from dataclasses import dataclass
from typing import List
import re


@dataclass
class InputExample:
    question: str
    answer: List[str]


def byte_str(text):
    return text.encode('utf-8')


def byte_slice(byte_text, start, end):
    return byte_text[start: end].decode('utf-8')


def get_examples(qas):
    examples = []
    for q in tqdm(qas):
        plane = q['document_plaintext']
        question = q['question_text']
        answer = []

        byte_plane = byte_str(plane)

        for ans in q['passage_answer_candidates']:
            answer.append(byte_slice(byte_plane, ans['plaintext_start_byte'], ans['plaintext_end_byte']))

        examples.append(InputExample(question=question, answer=answer))

    return examples


def create_collections_from_examples(examples):
    collections = []
    queries = []

    data_format = "{}\t{}"
    collection_id = 0

    for example in examples:
        answer_pid = []
        for answer in example.answer:
            pattern = '[\r|\n|\t]'
            answer = re.sub(pattern=pattern, repl=' ', string=answer)
            collections.append(data_format.format(collection_id, answer))
            answer_pid.append(collection_id)
            collection_id += 1

        queries.append({
            "query": example.question,
            "answers": answer_pid
        })

    return collections, queries


def save_dataset(output_dir, collections, queries):
    with open(os.path.join(output_dir, "collection.tsv"), "w", encoding='utf-8') as f:
        for collection in collections:
            f.write(collection + "\n")

    with open(os.path.join(output_dir, "queries_answer.jsonl"), "w", encoding='utf-8') as f:
        for query in queries:
            f.write(json.dumps(query) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tydi_file", type=str)

    args = parser.parse_args()

    qas = []
    with jsonlines.open(args.tydi_file, "r") as f:
        for line in tqdm(f):
            if line['language'] == 'korean':
                qas.append(line)

    examples = get_examples(qas)
    dev_collections, dev_queries = create_collections_from_examples(examples)
    save_dataset("data/tydi", dev_collections, dev_queries)