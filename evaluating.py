from collections import Counter
from utils import flatten_lists


# 用于评价模型 计算精确率 召回率 f1值
class Matrics(object):

    def __init__(self, golden_tags, predict_tags, remove_O=False):
        '''

        :param golden_tags:
        :param prdict_tags:
        :param remove_O:
        '''
        self.golden_tags = flatten_lists(golden_tags)
        self.predict_tags = flatten_lists(predict_tags)

        #
        if remove_O:
            self._remove_Otags()
        self.tagset = set(self.golden_tags)
        self.correct_tags_number = self.count_correct_tags()
        self.predict_tags_counter = Counter(self.predict_tags)
        self.golden_tags_counter = Counter(self.golden_tags)
        self.precision_scores = self.cal_precision()
        self.recall_scores = self.cal_recall()
        self.f1_score = self.cal_f1()

    # 去除所有O标记
    def _remove_Otags(self):
        length = len(self.golden_tags)
        O_tag_indices = [i for i in range(length) if self.golden_tags[i] == 'O']
        self.golden_tags = [tag for i, tag in enumerate(self.golden_tags) if i not in O_tag_indices]
        self.predict_tags = [tag for i, tag in enumerate(self.predict_tags) if i not in O_tag_indices]

        print("total tags {}, remove 'O' tags {}, {:.2f}% of total".format(
            length, len(O_tag_indices), len(O_tag_indices) / length * 100.0
        ))

    # 计算每种标签预测正确的个数 用于精确率和召回率的计算
    def count_correct_tags(self):
        correct_dict = {}
        for gold_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            if gold_tag == predict_tag:
                if gold_tag not in correct_dict:
                    correct_dict[gold_tag] = 1
                else:
                    correct_dict[gold_tag] += 1

        return correct_dict

    # 计算精确率
    def cal_precision(self):
        precision_scores = {}
        for tag in self.tagset:
            precision_scores[tag] = self.correct_tags_number.get(tag, 0) / self.predict_tags_counter[tag]

        return precision_scores

    # 计算召回率
    def cal_recall(self):
        recal_scores = {}
        for tag in self.tagset:
            recal_scores[tag] = self.correct_tags_number.get(tag, 0) / self.golden_tags_counter[tag]

        return recal_scores

    # 计算f1值
    def cal_f1(self):
        f1_scores = {}
        for tag in self.tagset:
            p, r = self.precision_scores[tag], self.recall_scores[tag]
            f1_scores[tag] = 2 * p * r / (p + r + 1e-10)
        return f1_scores
