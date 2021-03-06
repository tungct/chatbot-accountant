from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string
from src.io_utils import Io_Utils
from sklearn.metrics.pairwise import cosine_similarity
import setting

stop_words = [
            'tôi', 'em', 'chị', 'mình', 'với', 'vậy', 'như thế nào', 'không', 'thế nào',
            'thì', 'muốn', 'mà', 'để', 'phải', 'hãy', 'đang', 'cần', 'và', 'có', 'tại sao',
            'nhỉ', 'là', 'nên', 'ơi', 'bạn', 'giúp', 'tại', 'thế', 'nào', 'như', 'gì', 'sao',
            'à', 'thể', 'nhiên', 'tuy'
        ]

class Data_Utils():
    def __init__(self):
        self.TfidfVec = TfidfVectorizer()
        self.io_utils = Io_Utils()
        self.sent_tokens, self.labels = self.load_data(setting.TRAIN_PATH)
        #if os.path.exists(setting.TFIDF_DOC_PATH) == False:
        # self.tfidf = self.process_data()

    def map_qa(self):
        xl = self.io_utils.load_xlsx(setting.QA_PATH)
        qa = {
            0: 'Chúng tôi sẽ trả lời bạn sau',
            -1: 'Xin chào quý khách! Tôi có thể giúp gì ạ?'
        }
        for idx in range(len(xl)):
            stt, answer = xl['stt'][idx], xl['answer'][idx]
            qa[int(stt)] = answer
        return qa

    def load_data(self, path_data):
        with open(path_data, 'r', encoding='utf8', errors='ignore') as f:
            raw = f.readlines()
        sent_tokens, labels = [], []
        for i in range(len(raw)):
            data = raw[i].strip()
            if data != '':
                label = data.split(' ')[0].replace('###', '')
                sent = data.split('###' + label)[1]
                sent_tokens.append(sent)
                labels.append(int(label))
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

    def process_data(self, sent_tokens, labels):
        self.sent_tokens = sent_tokens
        self.labels = labels
        for idx in range(len(sent_tokens)):
            sent = sent_tokens[idx]
            sent = self.normalize_sentence(sent)
            sent_tokens[idx] = sent
        self.tfidf = self.TfidfVec.fit_transform(sent_tokens)


        # self.io_utils.save_pickle(self.TfidfVec, setting.TFIDF_VEC_PATH)
        # self.io_utils.save_pickle(self.tfidf, setting.TFIDF_DOC_PATH)
        #
        # self.io_utils.save_pickle(self.CountVec, setting.COUNT_VEC_PATH)
        # self.io_utils.save_pickle(self.count, setting.COUNT_DOC_PATH)
        return self.tfidf, self.TfidfVec

if __name__ == '__main__':
    data_utils = Data_Utils()
    io_utils = Io_Utils()
    new_doc = 'giá sản phẩm như nào thế'
    new_doc = data_utils.normalize_sentence(new_doc)
    print(new_doc)
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
    print(data_utils.labels[idx])
    print(data_utils.sent_tokens[idx])

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


