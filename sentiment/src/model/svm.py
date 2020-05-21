from sklearn.svm import LinearSVC
from sentiment.src.utils import load_conf
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
import unidecode

conf = load_conf.get_conf()
svm_conf = load_conf.get_svm_conf(conf)

class SVM:
    def __init__(self):
        self.estimator_C = svm_conf.get('estimator_C')
        self.lower_tfidf__ngram_range = svm_conf.get('lower_tfidf__ngram_range')
        self.with_tone_char__ngram_range = svm_conf.get('with_tone_char__ngram_range')
        self.remove_tone__tfidf__ngram_range = svm_conf.get('remove_tone__tfidf__ngram_range')
        self.min_df_word = svm_conf.get('min_df_word')
        self.min_df_char = svm_conf.get('min_df_char')
        self.analyzer = svm_conf.get('analyzer')
        self.svm_model = self.create_model()

    def create_model(self):
        svm = Pipeline([
            ('features', FeatureUnion([
                ('union_tfidf', Pipeline([('tfidf_word', TfidfVectorizer(ngram_range=self.lower_tfidf__ngram_range, min_df=self.min_df_word))])),
                    ('tfidf_char', TfidfVectorizer(ngram_range=self.with_tone_char__ngram_range, min_df=self.min_df_char, analyzer=self.analyzer)),
                ])),
            ('clf', LinearSVC(C=self.estimator_C))
        ])
        return svm

    def fit(self, X, Y):
        self.svm_model.fit(X, Y)
        pass

    def predict(self, X):
        Y = self.svm_model.predict(X)
        return Y
