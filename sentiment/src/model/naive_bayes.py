from collections import Counter
from decimal import Decimal as D

class NaiveBayes:

    def __init__(self):
        self.len_vocab = 0
        self.ratio_each_labels = {}
        self.count_each_labels = {}
        self.sum_each_labels = {}

    def get_ratio_each_labels(self, labels):
        count_labels = dict(Counter(labels))
        ratio_labels = {label: D(float(count)) / total for total in (sum(count_labels.values()),) for label, count in count_labels.items()}

        return ratio_labels

    def get_calc_each_labels(self, sentences, labels):
        dict_count_labels = {label: [] for label in set(labels)}

        for i in range(len(labels)):
            label, sentence = labels[i], sentences[i]
            dict_count_labels[label] += sentence.split()

        dict_sum_labels = {label: len(list_word) for label, list_word in dict_count_labels.items()}

        for label, word_list in dict_count_labels.items():
            count_word = dict(Counter(word_list))
            dict_count_labels[label] = count_word

        return dict_count_labels, dict_sum_labels

    def fit(self, sentences, labels):
        self.ratio_each_labels = self.get_ratio_each_labels(labels)
        self.count_each_labels, self.sum_each_labels = self.get_calc_each_labels(sentences, labels)

        for label, sum_word in self.sum_each_labels.items():
            self.len_vocab += sum_word
        all_word = []
        for sentence in sentences:
            all_word += sentence.split()
        self.len_vocab = len(set(all_word))

        pass

    def predict(self, X):
        y_pred = []
        for x in X:
            x = x.split()
            prob_words_in_label, prob_label = {}, {}
            for label, ratio in self.ratio_each_labels.items():
                prob_words_in_label[label] = D(1.0)
                for word in x:
                    prob_words_in_label[label] *= D(float(self.count_each_labels[label].get(word, 0) + 1)) / D((self.sum_each_labels[label] + self.len_vocab))
                prob_label[label] = D(ratio) * D(prob_words_in_label[label])
            label_predict = max(prob_label, key=prob_label.get)

            y_pred.append(label_predict)
        return y_pred


if __name__ == '__main__':
    labels = ['1', '1', '1', '2']
    sentences = ['Chinese Beijing Chinese', 'Chinese Chinese Shanghai', 'Chinese Macao', 'Tokyo Japan Chinese']
    test = ['Chinese Chinese Chinese Tokyo Japan', 'c']
    naive_bayes = NaiveBayes()
    naive_bayes.fit(sentences, labels)
    print(naive_bayes.len_vocab)
    print("sum_each_labels",naive_bayes.sum_each_labels)
    print("ratio_each_labels", naive_bayes.ratio_each_labels)
    print("count_each_labels", naive_bayes.count_each_labels)
    naive_bayes.predict(test)


