from utils import load_model
from evaluating import Matrics

HMM_MODEL_PATH = './ckpts/hmm.pkl'

def hmm_eval(train_data, test_data, word2id, tag2id, remove_O=False):
    test_word_lists, test_tag_lists = test_data

    hmm_model =  load_model(HMM_MODEL_PATH)
    pred_tag_lists = hmm_model.test(
        test_word_lists,
        word2id,
        tag2id
    )
    metrics = Matrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)


    # 计算各个数值与混淆矩阵
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists
