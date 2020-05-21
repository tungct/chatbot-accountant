import configparser

conf_dir = '../../configs/config.ini'

def get_conf():
    conf_file = conf_dir
    config = configparser.ConfigParser()
    config.read(conf_file)
    return config

def get_svm_conf(config):
    svm_conf = config['SVM_CONF']
    conf_param = {
        'estimator_C' : float(svm_conf['estimator_C']),
        'lower_tfidf__ngram_range' : eval(svm_conf['lower_tfidf__ngram_range']),
        'with_tone_char__ngram_range' : eval(svm_conf['with_tone_char__ngram_range']),
        'remove_tone__tfidf__ngram_range' : eval(svm_conf['remove_tone__tfidf__ngram_range']),
        'min_df_word' : int(svm_conf['min_df_word']),
        'min_df_char' : int(svm_conf['min_df_char']),
        'analyzer' : str(svm_conf['analyzer'])
    }
    return conf_param

if __name__ == '__main__':
    conf = get_conf()
    get_svm_conf(conf)





