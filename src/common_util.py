# -*- ecoding: utf-8 -*-
import numpy as np
from typing import List
import os
import shutil
import pickle
import json
import string



def data_generator(data, batch_size):
    assert batch_size > 0
    current_pos = 0
    index = 0
    total = len(data)//batch_size+1
    while current_pos < len(data):
        yield index, total , data[current_pos:current_pos + batch_size]
        current_pos += batch_size
        index += 1


def write2file(filename, text):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)


def write2file_arr(filename, arr):
    init = True
    with open(filename, 'w', encoding='utf-8') as f:
        for a in arr:
            if not init:
                f.write("\n")
                f.write(a)
            else:
                f.write(a)
                init = False


def readfile2arr(filename, skip=None):
    with open(filename, 'r', encoding='utf-8') as f:
        if skip == None:
            arr = [a.strip() for a in f]
        else:
            arr = [a.strip() for a in f][skip:]
    return arr


# 字符串最小编辑距离
def minDistance(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = [[0 for i in range(n + 1)] for j in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i]
                               [j - 1] + 1, dp[i - 1][j - 1] + 1)

    return dp[m][n]


def get_words2ids_pad(texts, words2id, max_seq):
    lines = []
    for line in texts:
        lines.append(line.strip())

    pad_ids = []
    for line in lines:
        ids = [words2id[i] for i in line]
        ids = ids[0:max_seq]
        if len(ids) < max_seq:
            ids = ids + [0] * (max_seq - len(ids))
        pad_ids.append(ids)
    return np.array(pad_ids)


def remove_dir_and_files(root_dir):
    # 列出该目录下的所有文件名
    shutil.rmtree(root_dir, True)


def save_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    if filename.endswith(".json"):
        return read_json(filename)
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


def arr2str_tab(arr):
    result = []
    for i in arr:
        if isinstance(i, int):
            result.append(str(i))
        elif isinstance(i, float):
            result.append("%.3f" % i)
        elif isinstance(i, str):
            result.append(i)
    # arr = [str(i) if isinstance(i, int) elif  "%.3f" % i for i in arr]
    return "\t".join(result)


def to_torch(data, type=None, device="cpu"):
    import torch
    if type == None:
        return torch.from_numpy(np.array(data)).to(device)
    else:
        return torch.from_numpy(np.array(data, dtype=type)).to(device)


def read_json(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        data = json.load(f)
        return data


def cached_pkl(cache_path: str, load_from_cache: bool, gen_func, *args, **kargs):
    if os.path.exists(cache_path) and load_from_cache:
        print("gen from cached pkl")
        return load_pickle(cache_path)
    else:
        print("gen from origin file")
        out = gen_func(*args, **kargs)
        save_pickle(cache_path, out)
        return out


def alignment(arr: List, max_length: int, padding):
    arr_len = len(arr)
    mask = [1.]*len(arr)
    if len(arr) >= max_length:
        arr = arr[0:max_length]
        mask = mask[0:max_length]
    else:
        arr += [padding]*(max_length-arr_len)
        mask += [0.]*(max_length-arr_len)
    return arr, mask


def collate_fn_from_map(data, keys = None, device="cpu", ignore_keys=[]):
    assert data
    assert isinstance(data, list)
    result = {}
    import torch

    ignore_keys = set(ignore_keys)
    if isinstance(data[0], dict):
        for i in data:
            for k, v in i.items():
                if k not in ignore_keys:
                    if k not in result:
                        result[k] = []
                    result[k].append(v)
        if keys:
            for k, v in result.items():
                if k in keys:
                    result[k] = to_torch(v, type=keys[k], device=device)

    return result


def save_json_datas(file:str, datas):
    # if not os.path.exists("../evaluate/python_result/"+testset_name+"/"):
        # os.mkdir("../evaluate/python_result/"+testset_name+"/")
    # paths=file.split("/")
    # tmp_path=paths[0]+"/"
    beg_index=0
    while True:
        try:
            index=file.index("/",beg_index)
        except Exception as e:
            break
        
        recur_dir_path=file[:index+1]
        if not os.path.exists(recur_dir_path):
            os.mkdir(recur_dir_path)
        beg_index=index+1

        
    with open(file, 'w', encoding="utf-8") as f:
        f.write(json.dumps(datas, indent=4, ensure_ascii=False))


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def save_huggface_json_datas(file, datas):
    line_datas = []
    for i in datas:
        line_datas.append(json.dumps(i, ensure_ascii=False))
    write2file_arr(file, line_datas)

def read_huggingface_json_datas(file):
    line_datas = []
    datas=readfile2arr(file)
    for i in datas:
        r=json.loads(i)
        line_datas.append(r)
    return line_datas

def get_punctuations():
    punctuations = set(list(string.punctuation + '。！？”’“‘…·《》【】—-,、，'))
    return punctuations
    

def random_select_unrepeat(items, size):
    # 不重复
    selected = np.random.choice(
        items, replace=False, size=size)
    return selected

# 将文档拆分成句子，根据max_len合并
def cut_to_sents(text,max_len=120,merge=False):
    import re
    sentences = re.split(r"(？|\?|。|！|\…\…|\r|\n)", text)
    
    clip_sents=[]
    for i in sentences:
        left=i
        while len(left)>0:
            clip_sents.append(left[:max_len])
            left=left[max_len:]
    return clip_sents

    for i in range(len(sentences)):
        if i%2==1:
            _sentences.append(sentences[i-1]+sentences[i])
        elif i==len(sentences)-1:
            # 最后一个
            _sentences.append(sentences[i])
    
    clip_sents=[]
    for i in _sentences:
        left=i
        while len(left)>0:
            clip_sents.append(left[:max_len])
            left=left[max_len:]
    #  不拼接
    if not merge:
        return clip_sents
    
    l=""
    merged_sents=[]
    for index,i in enumerate(clip_sents):
        if len(l)+len(i)>max_len:
            merged_sents.append(l)
            l=""
            l+=i
        else:
            l+=i

    if len(l)>0:
        merged_sents.append(l)
    return merged_sents

if __name__ == '__main__':
    # articles = readfile2arr("./datas/corpus/raw_articles.csv")
    # print(len(articles))
    # print(collate_fn_from_map(
    #     [{'a': 'texta', 'b': 'textb'}, {'a': 'texta2', 'b': 'textb2'}], keys=[]))
    data = {"a_f": "a", 'b_f': 'b'}
    datas = [data for i in range(10)]
    # print(json.dumps(data, ensure_ascii=False))
    # print(json.dumps(data, indent=4, ensure_ascii=False))
    save_huggface_json_datas("./test.json", datas)
