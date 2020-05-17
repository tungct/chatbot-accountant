import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = DIR_PATH + '/data/'
TRAIN_PATH = DATA_PATH + 'train.txt'
TEST_PATH = DATA_PATH + 'test.txt'

QA_PATH = DATA_PATH + 'chatbot-accountant-qa.xlsx'

TFIDF_VEC_PATH = DATA_PATH + 'tfidf_vec.pickle'
TFIDF_DOC_PATH = DATA_PATH + 'tfidf_docs.pickle'

COUNT_VEC_PATH = DATA_PATH + 'count_vec.pickle'
COUNT_DOC_PATH = DATA_PATH + 'count_docs.pickle'

