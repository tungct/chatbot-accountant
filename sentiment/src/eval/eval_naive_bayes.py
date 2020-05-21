import sys
sys.path.insert(0, '../../../')
sys.path.insert(0, '../')
from sentiment.src.model.naive_bayes import NaiveBayes
from sentiment.src.utils import pre_process, load_conf, io
from sentiment.src.eval import eval_utils
import numpy as np

conf = load_conf.get_conf()
svm_conf = load_conf.get_svm_conf(conf)
train_file = conf['IO']['train_dir']
test_file = conf['IO']['test_dir']
model_dir = conf['IO']['model_naive_bayes']

if __name__ == '__main__':
    sent_train, label_train = pre_process.read_data_src(train_file)
    X_train, Y_train = sent_train, np.array(label_train)

    sent_test, label_test = pre_process.read_data_src(test_file)
    X_test, Y_test = sent_test, np.array(label_test)

    clf = NaiveBayes()
    clf.fit(X_train, Y_train)
    io.save_model(clf, model_dir)

    Y_pred = clf.predict(X_test)

    class_names = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'])

    eval_utils.eval_model(y_true=Y_test, y_pred=Y_pred, class_names=class_names)