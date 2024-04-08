import os

from loguru import logger


class Report:
    def __init__(self):
        self.results = {}
        # dataset: {target: [ref_words_len, can_words_len, acc_word_len]}

    def compare_line(self, ref_words, can_words):  # reference 标注
        """
        Compare the reference and candidate line by line.

        Args:
            ref_words (list): The reference list.
            can_words (list): The candidate list.

        Returns:
            acc_word_len (int): The number of the correct words.

        """
        ref_words_len = len(ref_words)
        can_words_len = len(can_words)

        ref_index = []
        index = 0
        for word in ref_words:
            word_index = [index]
            index += len(word)
            word_index.append(index)
            ref_index.append(word_index)

        can_index = []
        index = 0
        for word in can_words:
            word_index = [index]
            index += len(word)
            word_index.append(index)
            can_index.append(word_index)

        tmp = [val for val in ref_index if val in can_index]
        acc_word_len = len(tmp)

        return ref_words_len, can_words_len, acc_word_len

    def add_result(self, dataset, target, ref_words_len, can_words_len, acc_word_len):
        if dataset not in self.results:
            self.results[dataset] = {}
        if target not in self.results[dataset]:
            self.results[dataset][target] = [ref_words_len, can_words_len, acc_word_len]
        else:
            self.results[dataset][target][0] += ref_words_len
            self.results[dataset][target][1] += can_words_len
            self.results[dataset][target][2] += acc_word_len

    def get_results(self):
        return self.results

    def output_results(self):
        for dataset, targets in self.results.items():
            logger.info(f"dataset: {dataset}")
            for target, result in targets.items():
                logger.info(f"target: {target}")
                ref_words_len, can_words_len, acc_word_len = result
                recall = self.cal_recall(ref_words_len, acc_word_len)
                precision = self.cal_precision(can_words_len, acc_word_len)
                f1 = self.cal_f1(recall, precision)
                logger.info(f"recall: {recall}")
                logger.info(f"precision: {precision}")
                logger.info(f"f1: {f1}")

    def cal_recall(self, ref_words_len, acc_word_len):
        return acc_word_len / ref_words_len

    def cal_precision(self, can_words_len, acc_word_len):
        return acc_word_len / can_words_len

    def cal_f1(self, recall, precision):
        return 2 * recall * precision / (recall + precision)
