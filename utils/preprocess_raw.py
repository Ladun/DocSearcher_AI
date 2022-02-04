
import argparse
import re
from glob import glob
from tqdm import tqdm
import os


def _cleansing(text):
    # email 제거
    pattern = '([a-zA-Z0-9\_.+-]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-.]+)'
    text = re.sub(pattern=pattern, repl='', string=text)
    # url 제거
    pattern = '(?:https?|ftp|file)://(?:[a-zA-Z]|[0-9]|[$\-@\.&+:/?=]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(pattern=pattern, repl='', string=text)
    # html 태그 제거
    pattern = '<[^>]*>'
    text = re.sub(pattern=pattern, repl='', string=text)
    # \r, \n 제거
    pattern = '[\r|\n]'
    text = re.sub(pattern=pattern, repl='', string=text)
    # 특수기호 제거
    pattern = '[^\w\s.]'
    text = re.sub(pattern=pattern, repl='', string=text)
    # 한자 제거
    pattern = '[一-龥]'
    text = re.sub(pattern=pattern, repl='', string=text)
    # 이중 space 제거
    pattern = re.compile(r'\s{2,}')
    text = re.sub(pattern=pattern, repl='', string=text)

    return text


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
