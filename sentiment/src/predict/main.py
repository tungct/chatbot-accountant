import sys
sys.path.insert(0, '../../../')
sys.path.insert(0, '../')
from sentiment.src.utils import load_conf, io, pre_process

conf = load_conf.get_conf()
svm_model_dir = conf['IO']['model_svm']
naive_bayes_model_dir = conf['IO']['model_naive_bayes']
map_label = {
    1 : 'Positive',
    2 : 'Negative',
    3 : 'Neutral'
}

if __name__ == '__main__':
    sentence = str(sys.argv[1])
    # sentence = "Thiết kế khá đẹp đấy, ủng hộ hàng Việt nào ae eeee"
    sentence_norm = pre_process.seg_word_news(sentence).lower()

    naive_bayes_model = io.load_model(naive_bayes_model_dir)
    svm_model = io.load_model(svm_model_dir)

    nb_index_label_predict = naive_bayes_model.predict([sentence_norm])[0] + 1
    svm_index_label_predict = svm_model.predict([sentence_norm])[0] + 1

    nb_label_predict = map_label[nb_index_label_predict]
    svm_label_predict = map_label[svm_index_label_predict]

    print("Sentence:", sentence)
    print("Naive Bayes:", nb_label_predict)
    print("SVM:", svm_label_predict)






