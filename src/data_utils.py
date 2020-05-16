from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string
from src.io_utils import Io_Utils
from sklearn.metrics.pairwise import cosine_similarity
import setting
import os.path
stop_words = [
    'tôi', 'em', 'chị', 'mình', 'với', 'vậy', 'như thế nào', 'không', 'thế nào',
    'thì', 'muốn', 'mà', 'để', 'phải', 'hãy', 'đang', 'cần', 'và', 'có', 'tại sao',
    'nhỉ', 'là', 'nên', 'ơi', 'bạn', 'giúp', 'tại', 'thế', 'nào', 'như'
]

class Data_Utils():
    def __init__(self):
        self.TfidfVec = TfidfVectorizer()
        self.CountVec = CountVectorizer()
        self.io_utils = Io_Utils()
        self.sent_tokens, self.labels = self.load_data()
        #if os.path.exists(setting.TFIDF_DOC_PATH) == False:
        self.tfidf = self.process_data()

    def load_data(self):
        with open(setting.TRAIN_PATH, 'r', encoding='utf8', errors='ignore') as f:
            raw = f.readlines()
        sent_tokens, labels = [], []
        for i in range(len(raw)):
            data = raw[i].strip()
            if data != '':
                label = data.split(' ')[0].replace('###', '')
                sent = data.split('###' + label)[1]
                sent_tokens.append(sent)
                labels.append(label)
        return sent_tokens, labels

    def normalize_sentence(self, sentence):
        sentence = sentence.lower()
        for punc in string.punctuation:
            if punc in sentence:
                sentence = sentence.replace(punc, ' ')
        sentences = sentence.split()
        for word in stop_words:
            if word in sentences:
                sentences.remove(word)
        sentence = ' '.join(sentences)
        return sentence

    def process_data(self):
        for idx in range(len(self.sent_tokens)):
            sent = self.sent_tokens[idx]
            sent = self.normalize_sentence(sent)
            self.sent_tokens[idx] = sent
        self.tfidf = self.TfidfVec.fit_transform(self.sent_tokens)
        self.count = self.CountVec.fit_transform(self.sent_tokens)

        self.io_utils.save_pickle(self.TfidfVec, setting.TFIDF_VEC_PATH)
        self.io_utils.save_pickle(self.tfidf, setting.TFIDF_DOC_PATH)

        self.io_utils.save_pickle(self.CountVec, setting.COUNT_VEC_PATH)
        self.io_utils.save_pickle(self.count, setting.COUNT_DOC_PATH)
        return self.count

if __name__ == '__main__':
    data_utils = Data_Utils()
    io_utils = Io_Utils()
    new_doc = 'Hướng dẫn tôi cách nhập kho vật tư'
    new_doc = data_utils.normalize_sentence(new_doc)
    tfidf_vec = io_utils.load_pickle(setting.TFIDF_VEC_PATH)
    tfidf_doc = io_utils.load_pickle(setting.TFIDF_DOC_PATH)

    new_vec = tfidf_vec.transform([new_doc]).toarray()

    vals = cosine_similarity(new_vec, tfidf_doc)
    idx = vals.argsort()[0][-1]
    idx_2 = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    print("TFIDF")
    req_tfidf = flat[-1]
    print(req_tfidf)
    print(data_utils.sent_tokens[idx])
    print(data_utils.labels[idx])
    req_tfidf = flat[-2]
    print(req_tfidf)
    print(data_utils.sent_tokens[idx_2])


    count_vec = io_utils.load_pickle(setting.COUNT_VEC_PATH)
    count_doc = io_utils.load_pickle(setting.COUNT_DOC_PATH)

    new_vec = count_vec.transform([new_doc]).toarray()

    vals = cosine_similarity(new_vec, count_doc)

    idx = vals.argsort()[0][-1]
    idx_2 = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    print("BOW")
    req_tfidf = flat[-1]
    print(req_tfidf)
    print(data_utils.sent_tokens[idx])
    req_tfidf = flat[-2]
    print(req_tfidf)
    print(data_utils.sent_tokens[idx_2])


