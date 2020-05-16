from src.io_utils import Io_Utils
from sklearn.metrics.pairwise import cosine_similarity
import setting
from src.data_utils import Data_Utils

class Clf_Utils():
    def __init__(self):
        self.stop_words = [
            'tôi', 'em', 'chị', 'mình', 'với', 'vậy', 'như thế nào', 'không', 'thế nào',
            'thì', 'muốn', 'mà', 'để', 'phải', 'hãy', 'đang', 'cần', 'và', 'có', 'tại sao',
            'nhỉ', 'là', 'nên', 'ơi', 'bạn', 'giúp', 'tại', 'thế', 'nào', 'như'
        ]
        self.data_utils = Data_Utils()
        self.io_utils = Io_Utils()
        self.tfidf_vec = self.io_utils.load_pickle(setting.TFIDF_VEC_PATH)
        self.tfidf_doc = self.io_utils.load_pickle(setting.TFIDF_DOC_PATH)

    def predict(self, sentences):
        labels = []
        for sentence in sentences:
            sentence = self.data_utils.normalize_sentence(sentence)
            new_vec = self.tfidf_vec.transform([sentence]).toarray()
            vals = cosine_similarity(new_vec, self.tfidf_doc)
            idx = vals.argsort()[0][-1]
            label = self.data_utils.labels[idx]
            labels.append(label)
        return labels

if __name__ == '__main__':
    sentences = ['Em ơi, ở máy trạm của bạn A, Tôi chỉ muốn bạn ý xem được dữ liệu HA, NB thôi, thì có được không']
    clf_utils = Clf_Utils()
    print(clf_utils.predict(sentences))


