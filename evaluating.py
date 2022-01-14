from collections import Counter
from utils import flatten_lists


# 用于评价模型 计算精确率 召回率 f1值
class Metrics(object):

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
        self.tagset = sorted(self.tagset)

        self.correct_tags_number = self.count_correct_tags()
        self.predict_tags_counter = Counter(self.predict_tags)
        self.golden_tags_counter = Counter(self.golden_tags)
        self.precision_scores = self.cal_precision()
        self.recall_scores = self.cal_recall()
        self.f1_scores = self.cal_f1()

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

    # 展示结果
    def report_scores(self):
        File = open('cache/eval.txt', 'w')
        print("           precision    recall  f1-score   support", file=File)
        row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}'

        for tag in self.tagset:
            print(row_format.format(
                tag, self.precision_scores[tag],
                self.recall_scores[tag],
                self.f1_scores[tag],
                self.golden_tags_counter[tag]
            ), file=File)

        avg_metrics = self._cal_weighted_average()
        print(row_format.format(
            'avg/total',
            avg_metrics['precision'],
            avg_metrics['recall'],
            avg_metrics['f1_score'],
            len(self.golden_tags)
        ), file=File)
        File.close()

    def _cal_weighted_average(self):
        weighted_average = {}
        total = len(self.golden_tags)

        # 计算weighted precisions:
        weighted_average['precision'] = 0.
        weighted_average['recall'] = 0.
        weighted_average['f1_score'] = 0.
        for tag in self.tagset:
            size = self.golden_tags_counter[tag]
            weighted_average['precision'] += self.precision_scores[tag] * size
            weighted_average['recall'] += self.recall_scores[tag] * size
            weighted_average['f1_score'] += self.f1_scores[tag] * size

        for metric in weighted_average.keys():
            weighted_average[metric] /= total

        return weighted_average

    # 计算混淆矩阵
    def report_confusion_matrix(self):
        File = open('cache/eval.txt', 'a+')
        print('\nConfusion Matrix:', file=File)
        tag_list = list(self.tagset)
        tags_size = len(tag_list)
        matrix = []
        # matrix[i][j] 表示第i个tag被模型预测称第j个tag的次数

        for i in range(tags_size):
            matrix.append([0] * tags_size)

        for golden_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            try:
                row = tag_list.index(golden_tag)
                col = tag_list.index(predict_tag)
                matrix[row][col] += 1
            except ValueError:
                # 未出现在golden_tags里的标记则跳过
                continue

        row_format_ = '{:>7} ' * (tags_size + 1)
        print(row_format_.format('', *tag_list), file=File)
        for i, row in enumerate(matrix):
            print(row_format_.format(tag_list[i], *row), file=File)

        File.close()