from data import build_corpus
from evaluate import hmm_train_eval, HMM_MODEL_PATH, crf_train_eval, bilstm_train_and_eval
from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from evaluating import Metrics


def main():
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    # HMM model
    # print("HMM model")

    # hmm_pred = hmm_train_eval(
    #     (train_word_lists, train_tag_lists),
    #     (test_word_lists, test_tag_lists),
    #     word2id,
    #     tag2id
    # )

    # CRF model
    # print("CRF model")
    # crf_model = crf_train_eval(
    #     (train_word_lists, train_tag_lists),
    #     (test_word_lists, test_tag_lists)
    # )

    print("Bi-LSTM+CRF Model")
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(train_word_lists, train_tag_lists)
    dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(dev_word_lists, dev_tag_lists)
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(test_word_lists, test_tag_lists, test=True)

    lstmcrf_pred = bilstm_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        crf_word2id, crf_tag2id
    )


if __name__ == '__main__':
    main()
