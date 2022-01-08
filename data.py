from os.path import join
from codecs import open

def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps


def build_corpus(split, make_vocab=True, data_dir='./ResumeNER'):
    '''
    读取数据
    :param split: train/dev/test 指定读取哪种文件
    :param make_vocab: 若为True，需多返回两个list：word2id和tag2id
    :param data_dir: 文件路径
    :return:
        word_lists: 读取的词语的list
        tag_lists: 读取的标记的list
        word2id: 词序列表，map格式
        tag2id: tag序列表，map格式
    '''

    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []

    # 读取文件 train/dev/test + .char.bmes
    # 将单词和标记序列分别放入word_lists和tag_lists中

    with open(join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []

        for line in f:
            if line != '\n' and line != '\r\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)

            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    # 如果make_vocab为True，需返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists