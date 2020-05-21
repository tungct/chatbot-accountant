import setting
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
count_vect = CountVectorizer()

stop_words = [
            'tôi', 'em', 'chị', 'mình', 'với', 'vậy', 'như thế nào', 'không', 'thế nào',
            'thì', 'muốn', 'mà', 'để', 'phải', 'hãy', 'đang', 'cần', 'và', 'có', 'tại sao',
            'nhỉ', 'là', 'nên', 'ơi', 'bạn', 'giúp', 'tại', 'thế', 'nào', 'như'
        ]

class Tranform_data:
    def read_data(self, path):
        with open(path, 'r') as file:
            data = file.readlines()
        return data

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

    def get_train_data(self, path_data):
        with open(path_data, 'r', encoding='utf8', errors='ignore') as f:
            raw = f.readlines()
        sent_tokens, labels = [], []
        for i in range(len(raw)):
            data = raw[i].strip()
            if data != '':
                label = data.split(' ')[0].replace('###', '')
                sent = data.split('###' + label)[1]
                sent = self.normalize_sentence(sent)
                sent_tokens.append(sent)
                labels.append(label)
        return sent_tokens, labels

    def conv_train_data(self, X, Y):
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(X)
        X = X.toarray()
        print(X.shape)
        vocab = vectorizer.get_feature_names()
        return X, Y, vectorizer

