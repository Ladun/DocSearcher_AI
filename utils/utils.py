
import numpy as np
import random
import re

import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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
    # []로 감싸진 숫자 제거
    pattern = '\[[\d]\]'
    text = re.sub(pattern=pattern, repl='', string=text)
    # \r, \n 제거
    pattern = '[\r|\n|\t]'
    text = re.sub(pattern=pattern, repl=' ', string=text)
    # 특수기호 제거
    pattern = '[^\w\s.]'
    text = re.sub(pattern=pattern, repl='', string=text)
    # 한자 제거
    pattern = '[一-龥]'
    text = re.sub(pattern=pattern, repl='', string=text)
    # 이중 space 제거
    pattern = re.compile(r'\s{2,}')
    text = re.sub(pattern=pattern, repl='', string=text)

    return text.strip()