from src.io_utils import Io_Utils
from sklearn.metrics.pairwise import cosine_similarity
import setting
from random import shuffle
from sklearn.model_selection import cross_val_score
from src.data_utils import Data_Utils
from sklearn.model_selection import StratifiedKFold
from statistics import mean
from sklearn.metrics import accuracy_score

class CS():
    def __init__(self):
        self.stop_words = [
            'tôi', 'em', 'chị', 'mình', 'với', 'vậy', 'như thế nào', 'không', 'thế nào',
            'thì', 'muốn', 'mà', 'để', 'phải', 'hãy', 'đang', 'cần', 'và', 'có', 'tại sao',
            'nhỉ', 'là', 'nên', 'ơi', 'bạn', 'giúp', 'tại', 'thế', 'nào', 'như', 'gì', 'sao',
            'à', 'thể', 'nhiên', 'tuy'
        ]
        self.data_utils = Data_Utils()
        self.io_utils = Io_Utils()
        # self.tfidf_vec = self.io_utils.load_pickle(setting.TFIDF_VEC_PATH)
        # self.tfidf_doc = self.io_utils.load_pickle(setting.TFIDF_DOC_PATH)
        self.map_qa = self.data_utils.map_qa()

    def fit(self, sentences, labels):
        self.sentences, self.labels = sentences, labels
        self.tfidf_doc, self.tfidf_vec = self.data_utils.process_data(sentences, labels)

    def predict(self, sentences):
        labels = []
        for sentence in sentences:
            sentence = self.data_utils.normalize_sentence(sentence)
            new_vec = self.tfidf_vec.transform([sentence]).toarray()
            vals = cosine_similarity(new_vec, self.tfidf_doc)
            flat = vals.flatten()
            flat.sort()
            req_tfidf = flat[-1]
            if req_tfidf > 0.35:
                idx = vals.argsort()[0][-1]
                label = self.data_utils.labels[idx]
                print(self.sentences[idx])
            else:
                label = 0
            labels.append(label)
        return labels

    def response_answer(self, cl_id):
        answer = self.map_qa[cl_id]
        return answer


if __name__ == '__main__':
    sentences = ['Tôi muốn phân bổ tự động tài sản cố định trên phần mềm thì phải nhập thế nào',
                 'Hướng dẫn Tôi cách nhập định mức nguyên vật liệu',
                 'Em ơi, ở máy trạm của bạn A, Tôi chỉ muốn bạn ý xem được dữ liệu HA, NB thôi, thì có được không'
                 ]
    clf = CS()
    X, y = clf.data_utils.sent_tokens, clf.data_utils.labels
    c = list(zip(X, y))

    shuffle(c)

    X, y = zip(*c)
    # scores = cross_val_score(clf, X, y, cv=5)
    kfold = StratifiedKFold(n_splits=7, shuffle=True, random_state=100)
    cvscores = []
    acc = []
    for train, test in kfold.split(X, y):
        X_train, y_train = [X[train[int(idx)]] for idx in range(len(train))], [y[train[int(idx)]] for idx in range(len(train))]
        X_test, y_test = [X[test[int(idx)]] for idx in range(len(test))], [y[test[int(idx)]] for idx in range(len(test))]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # for i in range(len(y_pred)):
        #     if y_pred[i] == 0:
        #         y_test[i] = 0
        print(X_test)
        print(y_pred)
        print(y_test)
        print(accuracy_score(y_test, y_pred))
        acc.append(accuracy_score(y_test, y_pred))
    print(mean(acc))

    # clf.fit(X, y)
    # y = clf.predict(sentences)
    # print(y)


