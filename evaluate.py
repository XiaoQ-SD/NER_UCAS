import time
from utils import load_model
from utils import save_model
from evaluating import Metrics
from models.HMM import HMM
from models.CRF import CRFModel
from models.bilstm_crf import BILSTM_Model

HMM_MODEL_PATH = './ckpts/hmm.pkl'

def hmm_train_eval(train_data, test_data, word2id, tag2id, remove_O=False):
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    hmm_model = HMM(len(tag2id), len(word2id))
    hmm_model.train(
        train_word_lists,
        train_tag_lists,
        word2id,
        tag2id
    )
    save_model(hmm_model, "./ckpts/hmm.pkl")

    pred_tag_lists = hmm_model.test(
        test_word_lists,
        word2id,
        tag2id
    )
    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    # 计算各个数值与混淆矩阵
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists

def crf_train_eval(train_data, test_data, remove_O = False):
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    crf_model = CRFModel()
    crf_model.train(train_word_lists, train_tag_lists)
    save_model(crf_model, "./ckpts/crf.pkl")

    pred_tag_lists = crf_model.test(test_word_lists)
    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()
    return pred_tag_lists

def bilstm_train_and_eval(train_data, dev_data, test_data,
                          word2id, tag2id, crf=True, remove_O=False):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstm_model = BILSTM_Model(vocab_size, out_size, crf=crf)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)

    model_name = "bilstm_crf" if crf else "bilstm"
    save_model(bilstm_model, "./ckpts/" + model_name + ".pkl")

    print("训练完毕,共用时{}秒.".format(int(time.time() - start)))
    print("评估{}模型中...".format(model_name))
    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists

