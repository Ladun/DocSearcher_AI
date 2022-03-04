
import argparse
import re
from glob import glob
from tqdm import tqdm
import os

from utils.utils import _cleansing


def preprocess_raw_data(lines):
    ret = []
    for line in tqdm(lines):
        text = _cleansing(line)
        if len(text) > 0:
            ret.append(text)
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--preprocess_dir", type=str, required=True)
    args = parser.parse_args()

    files = glob(f"{args.raw_dir}/*")

    if not os.path.exists(args.preprocess_dir):
        os.makedirs(args.preprocess_dir)

    for file_path in files:
        file_name = os.path.split(file_path)[-1]
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        print(f"--- Preprocessing {file_name} ---")
        preprocessed = preprocess_raw_data(lines)

        preprocessed_path = os.path.join(args.preprocess_dir, file_name)
        with open(preprocessed_path, "w", encoding='utf-8') as f:
            for line in preprocessed:
                f.write(line + '\n')

        print(f"Before len: {len(lines)}, After len: {len(preprocessed)}")


if __name__ == "__main__":
    main()
