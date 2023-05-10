from dataclasses import dataclass

import numpy as np
import metric
import common_util
# from sklearn.model_selection import KFold
import common_util
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelWithLMHead
import math
# from sklearn.metrics import f1_score, precision_score, recall_score
import os
from metric import EssayMetric
from corrector import NcnnCorrector, TorchCorrector
import time

if __name__ == "__main__":
    eval_files = [
        "../datasets/sighan_test_datas/sighan_test_13.json",
        "../datasets/sighan_test_datas/sighan_test_14.json",
        "../datasets/sighan_test_datas/sighan_test_15.json",
    ]
    # train_arg = Train_Arg()
    # bert_sighan_tokenizer = train_arg.tokenizer

    # # print(train_arg.task_name)
    # eval_result = []

    corrector = NcnnCorrector()

    # tokenizer = AutoTokenizer.from_pretrained("../models/tokenizer/")
    # bert_path="../models/hf/corrector_900news/"
    # model = AutoModelForMaskedLM.from_pretrained(bert_path)
    # corrector=TorchCorrector(model,tokenizer,"cpu")
    start_time = time.time()
    for eval_file in eval_files:
        testset_name = eval_file.split('/')[-1]
        testset_name = testset_name[:testset_name.index('.')]
        print("eval_file", eval_file)
        eval_datasets = common_util.read_json(eval_file)
        print(len(eval_datasets))
        # for model_path in model_paths:
        # model_name = model_path.split("/")[2]
        # print("###########################", "model_path",
        #   model_path, "###########################")
        #         print("data_count", len(eval_datasets))
        #         model = AutoModelWithLMHead.from_pretrained(model_path)
        #         model.to(train_arg.device)
        #         corrector = Corrector(
        #             model=model, tokenizer=train_arg.tokenizer, device=train_arg.device)

        essay_metric = EssayMetric()
        ########################
        pred_datas = []

        for index, i in enumerate(eval_datasets):
            # if index==10:
            #             #     break
            origin_text = i['origin_text']
            random_text = i['random_text']

            # pred_text, list_random_text, offset_mapping = corrector.correct(
            # random_text,return_list_origin_text=True)
            pred_text, list_random_text, offset_mapping = corrector.correct(
                random_text)

            origin_text = corrector._get_list_from_offset_mapping(
                origin_text, offset_mapping)
            random_text = list_random_text
            # print(pred_text)
            # break

            # sighan13 屏蔽掉 '的得地'
            if "spellgcn_test13" in eval_file:
                for j in range(len(random_text)):
                    if pred_text[j] == '的' or pred_text[j] == '得' or pred_text[
                            j] == '地':
                        pred_text[j] == random_text[j]
                        origin_text[j] == random_text[j]

            # print(origin_text)
            # print(pred_text)
            essay_metric.statistic(origin_text, random_text, pred_text)

            # src tgt pred
            pred_datas.append((random_text, origin_text, pred_text))

        result = essay_metric.result()
        # result['model_path'] = model_path
        result['eval_file'] = eval_file
        # print(result)

        # print(common_util.arr2str_tab(
        #     ['p', 'r', 'f', 'fpr', 'acc']))
        # print(common_util.arr2str_tab(
        #     [result['p'], result['r'], result['f'], result['fpr'], result['acc']]))
        print(
            common_util.arr2str_tab([
                'c_p', 'c_r', 'c_f', 's_d_p', 's_d_r', 's_d_f', 's_c_p',
                's_c_r', 's_c_f', 's_fpr', 's_acc', 'sentence_count'
            ]))
        print(
            common_util.arr2str_tab([
                result['p'], result['r'], result['f'], result['sentence_p'],
                result['sentence_r'], result['sentence_f'],
                result['sentence_c_p'], result['sentence_c_r'],
                result['sentence_c_f'], result['sentence_fpr'],
                result['sentence_acc'], result['sentence_count']
            ]))
    end_time = time.time()
    total_time_cost = end_time - start_time
    print(f"total time {total_time_cost}s")
    # eval_result.append(result)

    #         # print(essay_metric.sentence_TP,
    #         #     essay_metric.sentence_FP,
    #         #     essay_metric.sentence_TN,
    #         #     essay_metric.sentence_FN ,
    #         #     essay_metric.sentence_CD ,
    #         #     essay_metric.sentence_ED )

    # #         common_util.save_json_datas(
    # #             "../evaluate/python_result/"+model_name+"/"+testset_name+"/fp_cases.txt", essay_metric.fp_cases)
    # #         common_util.save_json_datas(
    # #             "../evaluate/python_result/"+model_name+"/"+testset_name+"/tp_cases.txt", essay_metric.tp_cases)
    # #         common_util.save_json_datas(
    # #             "../evaluate/python_result/"+model_name+"/"+testset_name+"/fn_cases.txt", essay_metric.fn_cases)
    # #         del corrector
    # #         del model
    # # common_util.save_json_datas(
    # #     "../evaluate/python_result/python_metric_result.json", eval_result)
