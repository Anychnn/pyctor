from dataclasses import dataclass
import numpy as np
import common_util
import torch
from torch import optim
import common_util
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelWithLMHead
import re
import math
import os
import string
from download_util import download
import onnxruntime
import onnx
import onnxruntime as rt


class NcnnCorrector:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("../models/tokenizer/")
        self.punctuation = string.punctuation + '。！？”’“‘…·《》【】—-'

        model_size = 126979102
        # print(os.path.getsize("/Users/bytedance/Documents/workspace/pyctor/models/ncnn/corrector.quant.onnx"))
        model_url = "https://huggingface.co/anyang/bert_chinese_corrector_ncnn/resolve/main/corrector.quant.onnx"
        # model_name = "corrector.quant.onnx"
        # model_name = "bert_fp32.onnx"
        model_name = "corrector.fp32.onnx"
        # model_name = "model_m.quant2.onnx"
        
        model_sha = "d9fc70641f6c938de203989dc819edb01e02259305b4c25b270ccafe41adde00"

        save_path = f"../models/ncnn/{model_name}"
        if not os.path.exists(save_path):
            result = download(model_url, save_path, model_sha, model_size)
            if not result:
                raise ValueError("Model download failed")

        self.sess = rt.InferenceSession(save_path,
                                        providers=['CPUExecutionProvider'])
        self.output_name = self.sess.get_outputs()[0].name

    def ignore_punctuation(self, list_origin_text, list_pred_text):
        punc = self.punctuation
        result = []
        for i in range(len(list_origin_text)):
            if list_pred_text[i] == None:
                result.append(list_origin_text[i])
            elif list_origin_text[i] in punc or list_pred_text[
                    i] in punc or list_pred_text[i] == '[UNK]':
                result.append(list_origin_text[i])
            else:
                result.append(list_pred_text[i])
        return result

    def _get_list_from_offset_mapping(self, text, offset_mapping):
        list_text = []
        for i in offset_mapping:
            start_offset = i[0]
            end_offset = i[1]
            if end_offset - start_offset > 0:
                list_text.append(text[start_offset:end_offset])
        return list_text

    def correct(self,
                text,
                return_list_origin_text=False,
                show_probs=False,
                ignore_puncatuation=True):
        tokenizer = self.tokenizer
        inputs_encoded = tokenizer(text, return_tensors="np")
        inputs_encoded = {k: v for k, v in inputs_encoded.items()}

        # result="".join(output)

        # return result
        # get offset_mapping to process input_list and output_list length mismatch erros like 123456788 -> 123456 ##78 ##8
        # input and output list should have same length in list,so i use offset_mapping to convert input_text to input_text_list
        # example:
        # 输入: 10~16岁看少儿接段的书，三楼则是4~9岁儿童借的区域
        # list_origin_text: ['10', '~', '16', '岁', '看', '少', '儿', '接', '段', '的', '书', '，', '三', '楼', '则', '是', '4', '~', '9', '岁', '儿', '童', '借', '的', '区', '域']
        offset_mapping = tokenizer(
            text, return_offsets_mapping=True)["offset_mapping"]
        list_origin_text = self._get_list_from_offset_mapping(
            text, offset_mapping)

        output = self.sess.run([self.output_name], inputs_encoded)
        list_pred_text = tokenizer.convert_ids_to_tokens(
            np.argmax(output[0], axis=-1)[0][1:-1])

        # logits = logits[:, 1:-1, :]
        # probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()

        # token_ids = np.argmax(probs, axis=-1)

        # list_pred_text = tokenizer.convert_ids_to_tokens(token_ids[0])
        # assert len(list_origin_text) == len(list_pred_text), "input_list and output_list must have same length, \ninput: " + \
        #     str(list_origin_text)+"  \noutput:"+str(list_pred_text)

        # if ignore_puncatuation:
        list_pred_text = self.ignore_punctuation(list_origin_text,
                                                 list_pred_text)

        for index, i in enumerate(list_pred_text):
            if i.startswith("##"):
                list_pred_text[index] = "".join(list(i)[2:])
            elif i.startswith("#"):
                list_pred_text[index] = "".join(list(i)[1:])

        # if return_list_origin_text:
        #     return list_pred_text, list_origin_text, offset_mapping
        return list_pred_text, list_origin_text, offset_mapping

    def correct_with_probs(self,
                           text,
                           return_list_origin_text=False,
                           show_probs=False,
                           ignore_puncatuation=True):
        self.model.eval()
        tokenizer = self.tokenizer
        text_inputs = tokenizer(text, return_tensors="pt")

        # get offset_mapping to process input_list and output_list length mismatch erros like 123456788 -> 123456 ##78 ##8
        # input and output list should have same length in list,so i use offset_mapping to convert input_text to input_text_list
        # example:
        # 输入: 10~16岁看少儿接段的书，三楼则是4~9岁儿童借的区域
        # list_origin_text: ['10', '~', '16', '岁', '看', '少', '儿', '接', '段', '的', '书', '，', '三', '楼', '则', '是', '4', '~', '9', '岁', '儿', '童', '借', '的', '区', '域']
        offset_mapping = tokenizer(
            text, return_offsets_mapping=True)["offset_mapping"]
        list_origin_text = self._get_list_from_offset_mapping(
            text, offset_mapping)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        outputs = self.model(**text_inputs)
        logits = outputs['logits']

        logits = logits[:, 1:-1, :]
        probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()
        token_ids = np.argmax(probs, axis=-1)

        #     token_ids_sort = np.argsort(probs, axis=-1)
        #     print(token_ids_sort.shape)
        #     print(token_ids_sort)
        sort_probs = np.sort(probs, axis=-1)[:, :, -1][0]
        # sort_probs = np.around(sort_probs, 8)
        # print(sort_probs)

        list_pred_text = tokenizer.convert_ids_to_tokens(token_ids[0])
        assert len(list_origin_text) == len(list_pred_text), "input_list and output_list must have same length, \ninput: " + \
            str(list_origin_text)+"  \noutput:"+str(list_pred_text)

        if ignore_puncatuation:
            list_pred_text = self.ignore_punctuation(list_origin_text,
                                                     list_pred_text)

        for index, i in enumerate(list_pred_text):
            if i.startswith("##"):
                list_pred_text[index] = "".join(list(i)[2:])
            elif i.startswith("#"):
                list_pred_text[index] = "".join(list(i)[1:])

        return list_pred_text, list_origin_text, offset_mapping, sort_probs

    def correct_long(self, text, ignore_puncatuation=True):
        list_pred_text = []
        list_random_text = []
        offset_mapping = []

        texts = []
        start_idx = 0
        total_len = len(text)
        seq_len = 510
        while (start_idx < total_len):
            texts.append(text[start_idx:start_idx + seq_len])
            start_idx += seq_len

        accum_offset = 0
        for t in texts:
            _list_pred_text, _list_random_text, _offset_mapping = self.correct(
                t,
                return_list_origin_text=True,
                ignore_puncatuation=ignore_puncatuation)
            _offset_mapping = [(i[0] + accum_offset, i[1] + accum_offset)
                               for i in _offset_mapping]
            list_pred_text.extend(_list_pred_text)
            list_random_text.extend(_list_random_text)
            offset_mapping.extend(_offset_mapping)
            accum_offset += len(t)

        return list_pred_text, list_random_text, offset_mapping

    def correct_filter_topp2(self, random_text, origin_text):
        # text_return, replace_count, retain_count, ground_truth_prob = corrector.correct_filter_topp(
        #         item['random_text'], item['origin_text'], p)

        self.model.eval()
        tokenizer = self.tokenizer
        random_inputs = tokenizer(random_text, return_tensors="pt")

        origin_ids = tokenizer(origin_text)['input_ids'][1:-1]

        # get offset_mapping to process input_list and output_list length mismatch erros like 123456788 -> 123456 ##78 ##8
        # input and output list should have same length in list,so i use offset_mapping to convert input_text to input_text_list
        # example:
        # 输入: 10~16岁看少儿接段的书，三楼则是4~9岁儿童借的区域
        # list_origin_text: ['10', '~', '16', '岁', '看', '少', '儿', '接', '段', '的', '书', '，', '三', '楼', '则', '是', '4', '~', '9', '岁', '儿', '童', '借', '的', '区', '域']
        offset_mapping = tokenizer(
            random_text, return_offsets_mapping=True)["offset_mapping"]
        list_origin_text = self._get_list_from_offset_mapping(
            origin_text, offset_mapping)

        list_random_text = self._get_list_from_offset_mapping(
            random_text, offset_mapping)

        text_inputs = {k: v.to(self.device) for k, v in random_inputs.items()}
        outputs = self.model(**text_inputs)
        logits = outputs['logits']

        logits = logits[:, 1:-1, :]

        probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()[0]

        token_ids = np.argmax(probs, axis=-1)

        sort_probs = np.sort(probs, axis=-1)[:, -1]

        list_pred_text = tokenizer.convert_ids_to_tokens(token_ids)
        assert len(list_origin_text) == len(list_pred_text), "input_list and output_list must have same length, \ninput: " + \
            str(list_origin_text)+"  \noutput:"+str(list_pred_text)

        list_pred_text = self.ignore_punctuation(list_origin_text,
                                                 list_pred_text)

        for index, i in enumerate(list_pred_text):
            if i.startswith("##"):
                list_pred_text[index] = "".join(list(i)[2:])
            elif i.startswith("#"):
                list_pred_text[index] = "".join(list(i)[1:])

        # list_probs=[]
        # for index,origin_id in enumerate(origin_ids):
        #     list_probs.append(probs[index,origin_id])

        # return list_random_text,list_origin_text,list_pred_text,list_probs

        return list_random_text, list_origin_text, list_pred_text, sort_probs

    def correct_filter_topp(self, random_text, origin_text, p):
        tokenizer = self.tokenizer
        text_inputs = tokenizer(random_text, return_tensors="pt")
        origin_ids = tokenizer(origin_text)['input_ids'][1:-1]

        # get offset_mapping to process input_list and output_list length mismatch erros like 123456788 -> 123456 ##78 ##8
        # input and output list should have same length in list,so i use offset_mapping to convert input_text to input_text_list
        # example:
        # 输入: 10~16岁看少儿接段的书，三楼则是4~9岁儿童借的区域
        # list_origin_text: ['10', '~', '16', '岁', '看', '少', '儿', '接', '段', '的', '书', '，', '三', '楼', '则', '是', '4', '~', '9', '岁', '儿', '童', '借', '的', '区', '域']
        offset_mapping = tokenizer(
            random_text, return_offsets_mapping=True)["offset_mapping"]
        list_random_text = self._get_list_from_offset_mapping(
            random_text, offset_mapping)

        list_origin_text = self._get_list_from_offset_mapping(
            origin_text, offset_mapping)

        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        outputs = self.model(**text_inputs)
        logits = outputs['logits']

        logits = logits[:, 1:-1, :]
        probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()[0]

        origin_probs = []

        assert len(list_random_text) == len(probs)

        filtered_random_text = []
        for index, origin_id in enumerate(origin_ids):

            if probs[index, origin_id] > p:
                filtered_random_text.append(list_random_text[index])
            else:
                filtered_random_text.append(list_origin_text[index])

        return "".join(filtered_random_text)

    # text_return, replace_count, retain_count, ground_truth_prob = corrector.correct_filter_topp(
    #         item['random_text'], item['origin_text'], p)


class TorchCorrector:

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.punctuation = string.punctuation + '。！？”’“‘…·《》【】—-'

    def ignore_punctuation(self, list_origin_text, list_pred_text):
        punc = self.punctuation
        result = []
        for i in range(len(list_origin_text)):
            if list_pred_text[i] == None:
                result.append(list_origin_text[i])
            elif list_origin_text[i] in punc or list_pred_text[
                    i] in punc or list_pred_text[i] == '[UNK]':
                result.append(list_origin_text[i])
            else:
                result.append(list_pred_text[i])
        return result

    def _get_list_from_offset_mapping(self, text, offset_mapping):
        list_text = []
        for i in offset_mapping:
            start_offset = i[0]
            end_offset = i[1]
            if end_offset - start_offset > 0:
                list_text.append(text[start_offset:end_offset])
        return list_text

    def correct(self,
                text,
                return_list_origin_text=False,
                show_probs=False,
                ignore_puncatuation=True):
        self.model.eval()
        tokenizer = self.tokenizer
        text_inputs = tokenizer(text, return_tensors="pt")

        # get offset_mapping to process input_list and output_list length mismatch erros like 123456788 -> 123456 ##78 ##8
        # input and output list should have same length in list,so i use offset_mapping to convert input_text to input_text_list
        # example:
        # 输入: 10~16岁看少儿接段的书，三楼则是4~9岁儿童借的区域
        # list_origin_text: ['10', '~', '16', '岁', '看', '少', '儿', '接', '段', '的', '书', '，', '三', '楼', '则', '是', '4', '~', '9', '岁', '儿', '童', '借', '的', '区', '域']
        offset_mapping = tokenizer(
            text, return_offsets_mapping=True)["offset_mapping"]
        list_origin_text = self._get_list_from_offset_mapping(
            text, offset_mapping)
        # for i in offset_mapping:
        #     start_offset = i[0]
        #     end_offset = i[1]
        #     if end_offset-start_offset > 0:
        #         list_origin_text.append(text[start_offset:end_offset])
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        outputs = self.model(**text_inputs)
        logits = outputs['logits']

        logits = logits[:, 1:-1, :]
        probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()

        token_ids = np.argmax(probs, axis=-1)

        if show_probs:
            token_ids_sort = np.argsort(probs, axis=-1)
            print(token_ids_sort.shape)
            print(token_ids_sort)
            sort_probs = np.sort(probs, axis=-1)[:, :, -1][0]
            sort_probs = np.around(sort_probs, 8)
            print(sort_probs)

        list_pred_text = tokenizer.convert_ids_to_tokens(token_ids[0])
        assert len(list_origin_text) == len(list_pred_text), "input_list and output_list must have same length, \ninput: " + \
            str(list_origin_text)+"  \noutput:"+str(list_pred_text)

        if ignore_puncatuation:
            list_pred_text = self.ignore_punctuation(list_origin_text,
                                                     list_pred_text)

        for index, i in enumerate(list_pred_text):
            if i.startswith("##"):
                list_pred_text[index] = "".join(list(i)[2:])
            elif i.startswith("#"):
                list_pred_text[index] = "".join(list(i)[1:])

        if return_list_origin_text:
            return list_pred_text, list_origin_text, offset_mapping
        return list_pred_text

    def correct_with_probs(self,
                           text,
                           return_list_origin_text=False,
                           show_probs=False,
                           ignore_puncatuation=True):
        self.model.eval()
        tokenizer = self.tokenizer
        text_inputs = tokenizer(text, return_tensors="pt")

        # get offset_mapping to process input_list and output_list length mismatch erros like 123456788 -> 123456 ##78 ##8
        # input and output list should have same length in list,so i use offset_mapping to convert input_text to input_text_list
        # example:
        # 输入: 10~16岁看少儿接段的书，三楼则是4~9岁儿童借的区域
        # list_origin_text: ['10', '~', '16', '岁', '看', '少', '儿', '接', '段', '的', '书', '，', '三', '楼', '则', '是', '4', '~', '9', '岁', '儿', '童', '借', '的', '区', '域']
        offset_mapping = tokenizer(
            text, return_offsets_mapping=True)["offset_mapping"]
        list_origin_text = self._get_list_from_offset_mapping(
            text, offset_mapping)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        outputs = self.model(**text_inputs)
        logits = outputs['logits']

        logits = logits[:, 1:-1, :]
        probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()

        token_ids = np.argmax(probs, axis=-1)

        #     token_ids_sort = np.argsort(probs, axis=-1)
        #     print(token_ids_sort.shape)
        #     print(token_ids_sort)
        sort_probs = np.sort(probs, axis=-1)[:, :, -1][0]
        # sort_probs = np.around(sort_probs, 8)
        # print(sort_probs)

        list_pred_text = tokenizer.convert_ids_to_tokens(token_ids[0])
        assert len(list_origin_text) == len(list_pred_text), "input_list and output_list must have same length, \ninput: " + \
            str(list_origin_text)+"  \noutput:"+str(list_pred_text)

        if ignore_puncatuation:
            list_pred_text = self.ignore_punctuation(list_origin_text,
                                                     list_pred_text)

        for index, i in enumerate(list_pred_text):
            if i.startswith("##"):
                list_pred_text[index] = "".join(list(i)[2:])
            elif i.startswith("#"):
                list_pred_text[index] = "".join(list(i)[1:])

        return list_pred_text, list_origin_text, offset_mapping, sort_probs

    def correct_long(self, text, ignore_puncatuation=True):
        list_pred_text = []
        list_random_text = []
        offset_mapping = []

        texts = []
        start_idx = 0
        total_len = len(text)
        seq_len = 510
        while (start_idx < total_len):
            texts.append(text[start_idx:start_idx + seq_len])
            start_idx += seq_len

        accum_offset = 0
        for t in texts:
            _list_pred_text, _list_random_text, _offset_mapping = self.correct(
                t,
                return_list_origin_text=True,
                ignore_puncatuation=ignore_puncatuation)
            _offset_mapping = [(i[0] + accum_offset, i[1] + accum_offset)
                               for i in _offset_mapping]
            list_pred_text.extend(_list_pred_text)
            list_random_text.extend(_list_random_text)
            offset_mapping.extend(_offset_mapping)
            accum_offset += len(t)

        return list_pred_text, list_random_text, offset_mapping

    def correct_filter_topp2(self, random_text, origin_text):
        # text_return, replace_count, retain_count, ground_truth_prob = corrector.correct_filter_topp(
        #         item['random_text'], item['origin_text'], p)

        self.model.eval()
        tokenizer = self.tokenizer
        random_inputs = tokenizer(random_text, return_tensors="pt")

        origin_ids = tokenizer(origin_text)['input_ids'][1:-1]

        # get offset_mapping to process input_list and output_list length mismatch erros like 123456788 -> 123456 ##78 ##8
        # input and output list should have same length in list,so i use offset_mapping to convert input_text to input_text_list
        # example:
        # 输入: 10~16岁看少儿接段的书，三楼则是4~9岁儿童借的区域
        # list_origin_text: ['10', '~', '16', '岁', '看', '少', '儿', '接', '段', '的', '书', '，', '三', '楼', '则', '是', '4', '~', '9', '岁', '儿', '童', '借', '的', '区', '域']
        offset_mapping = tokenizer(
            random_text, return_offsets_mapping=True)["offset_mapping"]
        list_origin_text = self._get_list_from_offset_mapping(
            origin_text, offset_mapping)

        list_random_text = self._get_list_from_offset_mapping(
            random_text, offset_mapping)

        text_inputs = {k: v.to(self.device) for k, v in random_inputs.items()}
        outputs = self.model(**text_inputs)
        logits = outputs['logits']

        logits = logits[:, 1:-1, :]

        probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()[0]

        token_ids = np.argmax(probs, axis=-1)

        sort_probs = np.sort(probs, axis=-1)[:, -1]

        list_pred_text = tokenizer.convert_ids_to_tokens(token_ids)
        assert len(list_origin_text) == len(list_pred_text), "input_list and output_list must have same length, \ninput: " + \
            str(list_origin_text)+"  \noutput:"+str(list_pred_text)

        list_pred_text = self.ignore_punctuation(list_origin_text,
                                                 list_pred_text)

        for index, i in enumerate(list_pred_text):
            if i.startswith("##"):
                list_pred_text[index] = "".join(list(i)[2:])
            elif i.startswith("#"):
                list_pred_text[index] = "".join(list(i)[1:])

        # list_probs=[]
        # for index,origin_id in enumerate(origin_ids):
        #     list_probs.append(probs[index,origin_id])

        # return list_random_text,list_origin_text,list_pred_text,list_probs

        return list_random_text, list_origin_text, list_pred_text, sort_probs

    def correct_filter_topp(self, random_text, origin_text, p):
        tokenizer = self.tokenizer
        text_inputs = tokenizer(random_text, return_tensors="pt")
        origin_ids = tokenizer(origin_text)['input_ids'][1:-1]

        # get offset_mapping to process input_list and output_list length mismatch erros like 123456788 -> 123456 ##78 ##8
        # input and output list should have same length in list,so i use offset_mapping to convert input_text to input_text_list
        # example:
        # 输入: 10~16岁看少儿接段的书，三楼则是4~9岁儿童借的区域
        # list_origin_text: ['10', '~', '16', '岁', '看', '少', '儿', '接', '段', '的', '书', '，', '三', '楼', '则', '是', '4', '~', '9', '岁', '儿', '童', '借', '的', '区', '域']
        offset_mapping = tokenizer(
            random_text, return_offsets_mapping=True)["offset_mapping"]
        list_random_text = self._get_list_from_offset_mapping(
            random_text, offset_mapping)

        list_origin_text = self._get_list_from_offset_mapping(
            origin_text, offset_mapping)

        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        outputs = self.model(**text_inputs)
        logits = outputs['logits']

        logits = logits[:, 1:-1, :]
        probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()[0]

        origin_probs = []

        assert len(list_random_text) == len(probs)

        filtered_random_text = []
        for index, origin_id in enumerate(origin_ids):

            if probs[index, origin_id] > p:
                filtered_random_text.append(list_random_text[index])
            else:
                filtered_random_text.append(list_origin_text[index])

        return "".join(filtered_random_text)

    # text_return, replace_count, retain_count, ground_truth_prob = corrector.correct_filter_topp(
    #         item['random_text'], item['origin_text'], p)


if __name__ == '__main__':

    model = None

    # tokenizer = None
    # corrector=Corrector(mode)
    # @dataclass
    # class train_arg:
    #     # bert_path = './bert-base-chinese/'
    #     # bert_path = '../backup/random_gen_10_with_origin_concate_10w/checkpoint-64400/'
    #     bert_path = '../../amlt_corrector/man_label_finetune_1/checkpoint-3600/'
    #     # bert_path = '../../amlt_corrector/output/sighan_with_wiki_pinyin/checkpoint-90000/'
    #     vocab_path = "../bert-base-chinese/"
    #     # bert_path = './output/e_0/'
    #     # device = "cuda" if torch.cuda.is_available() else "cpu"
    #     device = "cuda"
    #     max_length = 510
    #     tokenizer = AutoTokenizer.from_pretrained(vocab_path)
    #     out_path = "./output/"

    # model = AutoModelForMaskedLM.from_pretrained(train_arg.bert_path)
    # model.to(train_arg.device)
    device = "cpu"
    corrector = NcnnCorrector()

    text = "你好，很高性见到你!"
    result = corrector.correct(text)
    print(result)
    # text = "孙悟空的金[MASK]棒"
    # text = "10~16岁看少儿接段的书，三楼则是4~9岁儿童借的区域"
    # text = "孙悟空的金铜棒[SEP]孙悟空的武器是金箍棒"
    # text = "孙悟空的金铜棒"
    # print("输入:", text)
    # corrected_text = corrector.correct(text, show_probs=True)
    # print(corrected_text)
