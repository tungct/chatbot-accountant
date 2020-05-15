from sklearn.feature_extraction.text import TfidfVectorizer
import string
from src.io_utils import Io_Utils
from sklearn.metrics.pairwise import cosine_similarity
import setting
import os.path

class Data_Utils():
    def __init__(self):
        self.TfidfVec = TfidfVectorizer()
        self.io_utils = Io_Utils()
        self.sent_tokens = self.load_data()
        if os.path.exists(setting.TFIDF_DOC_PATH) == False:
            self.tfidf = self.process_data()

    def load_data(self):
        with open(setting.TRAIN_PATH, 'r', encoding='utf8', errors='ignore') as f:
            raw = f.readlines()
        sent_tokens = []
        for i in range(len(raw)):
            data = raw[i].strip()
            if data != '':
                sent_tokens.append(data)
        return sent_tokens

    def normalize_sentence(self, sentence):
        for punc in string.punctuation:
            if punc in sentence:
                sentence = sentence.replace(punc, ' ')
        sentence = ' '.join(sentence.split())
        return sentence

    def process_data(self):
        for idx in range(len(self.sent_tokens)):
            sent = self.sent_tokens[idx]
            sent = self.normalize_sentence(sent)
            self.sent_tokens[idx] = sent
        self.tfidf = self.TfidfVec.fit_transform(self.sent_tokens)
        self.io_utils.save_pickle(self.TfidfVec, setting.TFIDF_VEC_PATH)
        self.io_utils.save_pickle(self.tfidf, setting.TFIDF_DOC_PATH)
        return self.tfidf

if __name__ == '__main__':
    data_utils = Data_Utils()
    io_utils = Io_Utils()
    new_doc = 'Sao số liệu trong bảng cân đối phát sinh và trong bảng cân đối nhập xuất tồn lại bị lệch nhau?'
    new_doc = data_utils.normalize_sentence(new_doc)
    tfidf_vec = io_utils.load_pickle(setting.TFIDF_VEC_PATH)
    tfidf_doc = io_utils.load_pickle(setting.TFIDF_DOC_PATH)

    new_vec = tfidf_vec.transform([new_doc]).toarray()

    vals = cosine_similarity(new_vec, tfidf_doc)
    idx = vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    print(flat)
    req_tfidf = flat[-1]
    print(req_tfidf)
    print(data_utils.sent_tokens[idx])


