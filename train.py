from models.HMM import HMM
from utils import save_model

def hmm_train(train_data, test_data, word2id, tag2id):
    '''

    :param train_data:
    :param test_data:
    :return:
    '''
    train_word_lists, train_tag_lists = train_data
    # test_word_lists, test_tag_lists = test_data

    hmm_model = HMM(len(tag2id), len(word2id))
    hmm_model.train(
        train_word_lists,
        train_tag_lists,
        word2id,
        tag2id
    )
    save_model(hmm_model, "./ckpts/hmm.pkl")

