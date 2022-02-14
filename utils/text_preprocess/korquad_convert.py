
import argparse
import json
from tqdm import tqdm
from glob import glob
import random
import os

from utils.utils import _cleansing


class KorQuadExample:
    def __init__(self, query, pos, neg):
        self.query = query
        self.pos = pos
        self.neg = neg


def preprocessing_korquad(file, minimum_len):
    with open(file, "r") as f:
        korquad_json = json.load(f)

    processed = []
    for i, data in enumerate(tqdm(korquad_json['data'])):
        qas = data['qas']

        for j, q in enumerate(qas):
            query = q['question']
            pos = _cleansing(q['answer']['text'])
            if len(pos) >= minimum_len:
                processed.append((query, pos))

    return processed


def make_example(files, minimum_len):
    preprocessed_files = []
    for file in files:
        print(f"Load file and preprocessing {file}")
        preprocessed_files.append(preprocessing_korquad(file, minimum_len))

    examples = []
    for i in tqdm(range(0, len(preprocessed_files) - 1)):
        print(f"Make example[{i}][{len(preprocessed_files[i])}]")
        target_file = preprocessed_files[i + 1]
        target_idx = list(range(len(target_file)))
        random.shuffle(target_idx)

        cur_idx = 0
        for j, data in enumerate(preprocessed_files[i]):
            examples.append(KorQuadExample(query=data[0],
                                           pos=data[1],
                                           neg=target_file[target_idx[cur_idx]][1]))

            cur_idx = (cur_idx + 1) % len(target_idx)

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--minimum_len", type=int, default=0)

    args = parser.parse_args()

    if not os.path.exists(os.path.split(args.output_file)[0]):
        os.makedirs(os.path.split(args.output_file)[0])

    files = glob(args.input_dir + "/*")
    examples = make_example(files, args.minimum_len)

    store_list = []
    for example in tqdm(examples):
        store_list.append(f"{example.query}\t{example.pos}\t{example.neg}")

    print(f"Store list size = {len(store_list)}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for line in store_list:
            f.write(line + "\n")
    print(f"Save file to {args.output_file} success")


if __name__ == "__main__":
    main()


