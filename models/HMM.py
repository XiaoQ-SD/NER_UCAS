import torch


class HMM(object):
    '''
    状态转移概率矩阵：由某一个标注转移到下一个标注的概率
    观测概率矩阵：在某个标注下，生成某个词的概率
    初始状态分布：每一个标注作为句子第一个字的标注的概率
    '''

    def __init__(self, N, M):
        '''
        初始化函数
        :param N: 状态数，存在的标注种类
        :param M: 观测数，有多少不同的字

        A[i][j] 状态转移概率矩阵，从i状态转移到j状态的概率
        B[i][j] 观测概率矩阵，从i状态下生成j观测的概率
        Pi[i]   初始状态概率，初始时刻为状态i的概率
        '''
        self.N = N
        self.M = M

        self.A = torch.zeros(N, N)
        self.B = torch.zeros(N, M)
        self.Pi = torch.zeros(N)

    def train(self, word_lists, tag_lists, word2id, tag2id):
        '''
        HMM训练，根据训练语料对模型参数进行估计
        我们有观测序列以及其对应的状态序列，使用极大似然估计的方法估计隐马尔可夫模型的参数
        :param word_lists: 列表，每个元素由字组成
        :param tag_lists:  列表，每个元素由标注组成
        :param word2id:    字典，将字映射为ID
        :param tag2id:     字典，将标注映射为ID
        '''

        assert len(tag_lists) == len(word_lists)

        # 估计状态转移概率矩阵
        for tag_list in tag_lists:
            seq_len = len(tag_list)
            for i in range(seq_len - 1):
                current_tagid = tag2id[tag_list[i]]
                next_tagid = tag2id[tag_list[i + 1]]
                self.A[current_tagid][next_tagid] += 1

        # 将等于0的概率加上很小的数
        self.A[self.A == 0.] = 1e-10
        # 归一化
        self.A = self.A / self.A.sum(dim=1, keepdim=True)

        # 估计观测概率矩阵
        for tag_list, word_list in zip(tag_lists, word_lists):
            assert len(tag_list) == len(word_list)

            for tag, word in zip(tag_list, word_list):
                tag_id = tag2id[tag]
                word_id = word2id[word]
                self.B[tag_id][word_id] += 1

        self.B[self.B == 0.] = 1e-10
        self.B = self.B / self.B.sum(dim=1, keepdim=True)

        # 估计初始状态概率
        for tag_list in tag_lists:
            init_tagid = tag2id[tag_list[0]]
            self.Pi[init_tagid] += 1

        self.Pi[self.Pi == 0.] = 1e-10
        self.Pi = self.Pi / self.Pi.sum()

    def test(self, word_lists, word2id, tag2id):
        '''

        :param word_lists:
        :param word2id:
        :param tag2id:
        :return:
        '''

        pred_tag_lists = []
        for word_list in word_lists:
            pred_tag_list = self.decoding(word_list, word2id, tag2id)
            pred_tag_lists.append(pred_tag_list)
        return pred_tag_lists

    def decoding(self, word_list, word2id, tag2id):
        '''
        使用维特比算法对给定的观测序列求状态序列，即对字组成的序列，求对应的标注
        用动态规划解决马尔科夫模型预测的问题，求概率最大路径
        :param word_list: 传入字组成的序列
        :param word2id:
        :param tag2id:
        :return: 返回tagid序列
        '''

        # 很小的概率相乘可能造成下溢，故采用对数概率，很小的数被映射为很大的负数
        A = torch.log(self.A)
        B = torch.log(self.B)
        Pi = torch.log(self.Pi)

        # 维特比矩阵viterbi，维度为[状态数 序列长度]
        #
        # backpointer是与viterbi一样大小的矩阵
        # backpointer[i][j]存储j-1个标注的id，用以解码时回溯找到最优路径
        seq_len = len(word_list)
        viterbi = torch.zeros(self.N, seq_len)
        backpointer = torch.zeros(self.N, seq_len).long()

        # 第一步 初始化
        # Bt[word_id]表示字为word_id的时候，对应各个标记的概率
        start_wordid = word2id.get(word_list[0], None)
        Bt = B.t()
        if start_wordid is None:
            # 不在字典里则假设状态的概率分布是均匀的
            bt = torch.log(torch.ones(self.N) / self.N)
        else:
            bt = Bt[start_wordid]
        viterbi[:, 0] = Pi + bt
        backpointer[:, 0] = -1

        # 第二步 递推
        for step in range(1, seq_len):
            wordid = word2id.get(word_list[step], None)

            # bt指在t时刻字为wordid的概率分布
            # 字不在字典里 则假设状态的概率分布是均匀的
            if wordid is None:
                bt = torch.log(torch.ones(self.N) / self.N)
            else:
                bt = Bt[wordid]

            for tag_id in range(len(tag2id)):
                max_prob, max_id = torch.max(
                    viterbi[:, step - 1] + A[:, tag_id], dim=0
                )
                viterbi[tag_id, step] = max_prob + bt[tag_id]
                backpointer[tag_id, step] = max_id

        # 得到最大概率 即最优路径的概率
        best_path_prob, best_path_pointer = torch.max(
            viterbi[:, seq_len - 1], dim=0
        )

        # 回溯 求最优路径

        best_path_pointer = best_path_pointer.item()
        best_path = [best_path_pointer]
        for back_step in range(seq_len - 1, 0, -1):
            best_path_pointer = backpointer[best_path_pointer, back_step]
            best_path_pointer = best_path_pointer.item()
            best_path.append(best_path_pointer)

        assert len(best_path) == len(word_list)
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        tag_list = [id2tag[id_] for id_ in reversed(best_path)]

        return tag_list
