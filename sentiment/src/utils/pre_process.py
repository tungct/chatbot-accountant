from pyvi.pyvi import ViTokenizer
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
count_vect = CountVectorizer()

stop_words = [
    'tôi', 'em', 'chị', 'mình', 'với', 'vậy', 'như thế nào', 'không', 'thế nào',
    'thì', 'muốn', 'mà', 'để', 'phải', 'hãy', 'đang', 'cần', 'và', 'có', 'tại sao',
    'nhỉ', 'là', 'nên', 'ơi', 'bạn', 'giúp', 'tại', 'thế', 'nào', 'như'
]

def read_data(path):
    with open(path, 'r') as file:
        data = file.readlines()
    return data

def seg_word_news(news):
    """
    segment word
    :param news: news's content
    :return: segment content
    """
    seg_news = ViTokenizer.tokenize(news)
    return seg_news

def read_list_stop_word(file):
    with open(file, 'r', encoding="utf8", errors='ignore') as src_file:
        raw_data = src_file.readlines()
    for i in range(len(raw_data)):
        raw_data[i] = raw_data[i].replace('\n', '').replace(' ','_')
    return raw_data

def normalize_sentence(sentence):
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

def read_data_src(path_data):
    with open(path_data, 'r', encoding='utf8', errors='ignore') as f:
        raw = f.readlines()
    sent_tokens, labels = [], []
    for i in range(len(raw)):
        data = raw[i].strip()
        if data != '':
            label = data.split(' ')[0].replace('###', '')
            sent = data.split('###' + label)[1]
            sent = normalize_sentence(sent)
            sent_tokens.append(sent)
            labels.append(int(label))
    return sent_tokens, labels

def get_train_data(raw_file):
    data = read_data(raw_file)
    X, Y = [], []
    for i in range(len(data)):
        if i % 2 == 0:
            x = data[i].replace('\n', '').replace('.', ' ').split()
            x = ' '.join(x).lower()
            # x = seg_word_news(x).lower()
            X.append(x)
        else:
            y = data[i].replace('\n', '')
            Y.append(y)
    return X, Y

def get_test_data(raw_file):
    data = read_data(raw_file)
    X, Y = [], []
    for i in range(len(data)):
        x = data[i].replace('\n', '').replace('.', ' ').split()
        x = ' '.join(x).lower()
        # x = seg_word_news(x).lower()

        # x = ViTokenizer.tokenize(x)
        X.append(x)
    return X

def conv_train_data(X, Y):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(X)
    X = X.toarray()
    vocab = vectorizer.get_feature_names()
    return X, Y, vectorizer


