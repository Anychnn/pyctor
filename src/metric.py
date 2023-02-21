from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import string


class EssayMetric:
    def __init__(self):
        # char_level
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.CD = 0
        self.ED = 0
        self.y_true = []
        self.y_pred = []
        self.fp_cases = []
        self.fn_cases = []
        self.total_cases = []

        self.fp_cases = []
        self.fn_cases = []
        self.tp_cases = []
        self.fp_cases = []
        self.total_cases = []

        # sentence_level
        self.sentence_TP = 0
        self.sentence_FP = 0
        self.sentence_TN = 0
        self.sentence_FN = 0
        self.sentence_CD = 0
        self.sentence_ED = 0
        # self.tokenizer=tokenizer
        self.sentence_count = 0
        self.illegal_count = 0
        # self.punctuation =
        self.puncations = string.punctuation + '。！？”’“‘…·《》【】—-'+"".join(['”', '；', '、', '.', '，', '。', '-', '？', ',', '•', '～', '—',
                                                                          '：', '·', '··', '．', '...', '....', '——', '’', '！', '－－', '..', '"', '‘'])

    def increment_illegal_count(self):
        self.illegal_count += 1

    def statistic(self, target_essay: List, src_essay: List, pred_essay: List, only_puncation=False, only_words=True,idx=None):
        label = set()

        if only_puncation:
            filtered_src_essay = []
            filtered_target_essay = []
            filtered_pred_essay = []

            for index, i in enumerate(target_essay):
                if i in self.puncations:
                    filtered_src_essay.append(src_essay[index])
                    filtered_pred_essay.append(pred_essay[index])
                    filtered_target_essay.append(target_essay[index])

            src_essay = filtered_src_essay
            pred_essay = filtered_pred_essay
            target_essay = filtered_target_essay

        elif only_words:
            filtered_src_essay = []
            filtered_target_essay = []
            filtered_pred_essay = []

            for index, i in enumerate(target_essay):
                if i not in self.puncations:
                    filtered_src_essay.append(src_essay[index])
                    filtered_pred_essay.append(pred_essay[index])
                    filtered_target_essay.append(target_essay[index])

            src_essay = filtered_src_essay
            pred_essay = filtered_pred_essay
            target_essay = filtered_target_essay

        for i in range(len(target_essay)):
            if src_essay[i] == target_essay[i]:
                if pred_essay[i] == target_essay[i]:
                    self.TN += 1
                else:
                    self.FP += 1
                    label.add('FP')
            else:
                if pred_essay[i] == target_essay[i]:
                    self.TP += 1
                    self.CD += 1
                    label.add('TP')
                else:
                    if pred_essay[i] != target_essay[i] and pred_essay[i] != src_essay[i]:
                        self.ED += 1
                        self.TP += 1
                    else:
                        self.FN += 1
                    label.add('FN')

        self.total_cases.append({
            "orig_article": "".join(self.get_origin_show(target_essay, src_essay, pred_essay)),
            "pred_article": "".join(self.get_pred_show(target_essay, src_essay, pred_essay)),
            "label": ",".join(list(label))
        })

        # sentence_level statistic
        # 负样本 TN cases
        if src_essay == target_essay:
            # 原输入没有错别字，预测也没有错别字
            if pred_essay == target_essay:
                self.sentence_TN += 1
            # 原输入没有错别字，预测中出现了错别字
            else:
                # FP cases
                self.sentence_FP += 1
                self.fp_cases.append({
                    "idx":idx,
                    "orig_article": "".join(self.get_origin_show(target_essay, src_essay, pred_essay)),
                    "rand_article": "".join(self.get_src_show(target_essay, src_essay, pred_essay)),
                    "pred_article": "".join(self.get_pred_show(target_essay, src_essay, pred_essay))
                })
        # 正样本
        else:  # src != target 有错别字
            # 原输入出现错别字，所有的错别字都被正确纠正
            if target_essay == pred_essay:
                self.sentence_TP += 1
                self.sentence_CD += 1
                self.tp_cases.append({
                    "idx":idx,
                    "orig_article": "".join(self.get_origin_show(target_essay, src_essay, pred_essay)),
                    "rand_article": "".join(self.get_src_show(target_essay, src_essay, pred_essay)),
                    "pred_article": "".join(self.get_pred_show(target_essay, src_essay, pred_essay))
                })
            # 原输入出现错别字，所有的错别字的位置都识别正确，但没有都被正确纠正
            
            elif self._check_consistency(src_essay, pred_essay, target_essay):
                # FN in sentence correction-level
                # have spelling errors,all errors are detected, and corrected
                self.sentence_TP += 1
                self.sentence_ED += 1  
            else:  # 错别字没有检测出来
                # 原输入出现错别字，但是预测中没有全部地检测出错别字或者添加了一些非错别字的错误预测
                # FN cases, have spelling errors, but not all of them are detected or some are added
                self.sentence_FN += 1
                self.fn_cases.append({
                    "idx":idx,
                    "orig_article": "".join(self.get_origin_show(target_essay, src_essay, pred_essay)),
                    "rand_article": "".join(self.get_src_show(target_essay, src_essay, pred_essay)),
                    "pred_article": "".join(self.get_pred_show(target_essay, src_essay, pred_essay))
                })

        self.sentence_count += 1
        return ",".join(list(label))

    def _check_consistency(self, src_essay, pred_essay, target_essay):
        if len(src_essay) != len(pred_essay) or len(target_essay) != len(pred_essay):
            return False
        else:
            for i in range(len(src_essay)):
                if (src_essay[i] == target_essay[i] and src_essay[i] == pred_essay[i]) or (src_essay[i] != target_essay[i] and src_essay[i] != pred_essay[i]):
                    continue
                else:
                    # print(f"{src_essay[i]} {target_essay[i]} {pred_essay[i]}")
                    return False
            return True

    def get_src_show(self, true_essay, false_essay, pred_essay):
        result = []
        for i in range(len(true_essay)):
            result.append(false_essay[i])
        return result

    def get_origin_show(self, true_essay, false_essay, pred_essay):
        result = []
        for i in range(len(true_essay)):
            if true_essay[i] != false_essay[i]:
                result.append(false_essay[i])
                result.append('[')
                result.append(true_essay[i])
                result.append(']')
            else:
                result.append(false_essay[i])
        return result

    def get_pred_show(self, true_essay, false_essay, pred_essay):
        result = []
        for i in range(len(true_essay)):
            if true_essay[i] != pred_essay[i]:
                result.append('[')
                result.append(pred_essay[i])
                result.append(']')
            else:
                result.append(pred_essay[i])
        return result

    def result(self):
        # print("self.TP+self.FN",self.TP+self.FN)
        d_p = self.TP/(self.TP+self.FP) if (self.TP+self.FP) > 0 else 0
        d_r = self.TP/(self.TP+self.FN) if (self.TP+self.FN) > 0 else 0
        d_f = 2*d_p*d_r/(d_p+d_r) if (d_p+d_r) > 0 else 0
        c_p = self.CD/(self.TP+self.FP) if (self.TP+self.FP) > 0 else 0
        c_r = self.CD/(self.TP+self.FN) if (self.TP+self.FN) > 0 else 0
        c_f = 2*c_p*c_r/(c_p+c_r) if (c_p+c_r) > 0 else 0
        fpr = self.FP/(self.FP+self.TN) if (self.FP+self.TN) > 0 else -1
        acc = (self.TP+self.TN)/(self.TP+self.TN+self.FN+self.FP)
        # acc = accuracy_score(y_true=self.y_true, y_pred=self.y_pred)
        sentence_total = self.sentence_FP+self.sentence_FN + \
            self.sentence_TN+self.sentence_TP
        sentence_p = self.sentence_TP / \
            (self.sentence_TP+self.sentence_FP) if (self.sentence_TP +
                                                    self.sentence_FP) > 0 else 0
        sentence_r = self.sentence_TP / \
            (self.sentence_TP+self.sentence_FN) if (self.sentence_TP +
                                                    self.sentence_FN) > 0 else 0
        sentence_f = 2*sentence_p*sentence_r / \
            (sentence_p+sentence_r) if (sentence_p+sentence_r) > 0 else 0

        # sentence_c_p = self.sentence_CD / \
        #     (self.sentence_TP+self.sentence_FP) if (self.sentence_TP +
        #                                             self.sentence_FP) > 0 else 0
        # sentence_c_r = self.sentence_CD / \
        #     (self.sentence_TP+self.sentence_FN) if (self.sentence_TP +
        #                                             self.sentence_FN) > 0 else 0
        sentence_c_p = self.sentence_CD / \
            (self.sentence_CD+self.sentence_FP) if (self.sentence_CD +
                                                    self.sentence_FP) > 0 else 0
        sentence_c_r = self.sentence_CD / \
            (self.sentence_TP+self.sentence_FN) if (self.sentence_TP +
                                                    self.sentence_FN) > 0 else 0  # sentence_TP = sentence_CD + sentence_ED
        sentence_c_f = 2*sentence_c_p*sentence_c_r / \
            (sentence_c_p+sentence_c_r) if (sentence_c_p+sentence_c_r) > 0 else 0

        sentence_acc = (self.sentence_TN+self.sentence_TP)/sentence_total
        sentence_fpr = 0 if (
            self.sentence_FP+self.sentence_TN) == 0 else self.sentence_FP/(self.sentence_FP+self.sentence_TN)
        return {
            'p': d_p*100,
            'r': d_r*100,
            'f': d_f*100,
            'c_p': c_p*100,
            'c_r': c_r*100,
            'c_f': c_f*100,
            'fpr': fpr*100,
            'acc': acc*100,
            'illegal_count': self.illegal_count,
            'sentence_p': sentence_p*100,
            'sentence_r': sentence_r*100,
            'sentence_f': sentence_f*100,
            'sentence_c_p': sentence_c_p*100,
            'sentence_c_r': sentence_c_r*100,
            'sentence_c_f': sentence_c_f*100,
            'sentence_acc': sentence_acc*100,
            'sentence_fpr': sentence_fpr*100,
            'sentence_count': self.sentence_count
        }


if __name__ == '__main__':
    metric = EssayMetric()
    print(metric.result())
