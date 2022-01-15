import tkinter
import codecs
import tkinter.messagebox
import tkinter.font as tf
import tkinter as tk
import os
from data import build_corpus
from evaluate import hmm_train, hmm_eval, HMM_MODEL_PATH, CRF_MODEL_PATH, BiLSTMCRF_MODEL_PATH, crf_train, crf_eval, \
    bilstm_train, bilstm_eval
from utils import load_model, extend_maps, prepocess_data_for_lstmcrf, flatten_lists

HMM_Trained = False
CRF_Trained = False
BiLSTM_Trained = False

def deal(txt):
    txt = str(txt)
    txt = txt.replace(' ', '')
    txt = txt.replace('\n', '')
    txt = txt.replace('\r', '')
    txt = txt.replace('\t', '')

    path_solve = 'ResumeNER/dc.char.bmes'
    File = codecs.open(path_solve, 'w', 'utf-8')

    for i in txt:
        print(i + ' ' + 'O', file=File)
        if i == '。' or i == '！' or i == '？':
            print("", file=File)
    print('\n', end='', file=File)
    File.close()


def display():
    '''
    All Datas
    '''

    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    trwl, trtl, w2, t2 = build_corpus("train")
    dwl, dtl = build_corpus("dev", make_vocab=False)
    tewl, tetl = build_corpus("test", make_vocab=False)

    crf_word2id, crf_tag2id = extend_maps(w2, t2, for_crf=True)
    Btrain_word_lists, Btrain_tag_lists = prepocess_data_for_lstmcrf(trwl, trtl)
    Bdev_word_lists, Bdev_tag_lists = prepocess_data_for_lstmcrf(dwl, dtl)
    Btest_word_lists, Btest_tag_lists = prepocess_data_for_lstmcrf(tewl, tetl, test=True)

    def showeval():
        file_object = open("cache/eval.txt")
        file_context = file_object.read()
        e2.delete('1.0', 'end')
        e2.insert('end', file_context)

    def evalModel():
        global HMM_Trained
        global CRF_Trained
        global BiLSTM_Trained

        if len(lb.curselection()) == 1:
            value = lb.get(lb.curselection())
            if value == 'HMM':
                global HMM_Trained
                if HMM_Trained == False:
                    tkinter.messagebox.showinfo(title="error", message='please train first')
                    return
                hmm_eval(
                    (train_word_lists, train_tag_lists),
                    (test_word_lists, test_tag_lists),
                    word2id,
                    tag2id
                )
                showeval()
                tkinter.messagebox.showinfo(title="finished", message='evaluate finished')
            elif value == 'CRF':
                global CRF_Trained
                if CRF_Trained == False:
                    tkinter.messagebox.showinfo(title="error", message='please train first')
                    return
                crf_eval(
                    (train_word_lists, train_tag_lists),
                    (test_word_lists, test_tag_lists)
                )
                showeval()
                tkinter.messagebox.showinfo(title="finished", message='evaluate finished')
            elif value == 'BiLSTM':
                global BiLSTM_Trained
                if BiLSTM_Trained == False:
                    tkinter.messagebox.showinfo(title="error", message='please train first')
                    return
                bilstm_eval(
                    (Btrain_word_lists, Btrain_tag_lists),
                    (Bdev_word_lists, Bdev_tag_lists),
                    (Btest_word_lists, Btest_tag_lists),
                    crf_word2id, crf_tag2id
                )
                showeval()
                tkinter.messagebox.showinfo(title="finished", message='evaluate finished')
        else:
            tkinter.messagebox.showinfo(title='worning', message='please select one first')

    def trainModel():
        global HMM_Trained
        global CRF_Trained
        global BiLSTM_Trained

        if len(lb.curselection()) == 1:
            value = lb.get(lb.curselection())
            if value == 'HMM':
                global HMM_Trained
                if HMM_Trained == True:
                    return
                HMM_Trained = True
                hmm_train(
                    (train_word_lists, train_tag_lists),
                    (test_word_lists, test_tag_lists),
                    word2id,
                    tag2id
                )
                tkinter.messagebox.showinfo(title="finished", message='train finished')
            elif value == 'CRF':
                global CRF_Trained
                if CRF_Trained == True:
                    return
                CRF_Trained = True
                crf_train(
                    (train_word_lists, train_tag_lists),
                    (test_word_lists, test_tag_lists)
                )
                tkinter.messagebox.showinfo(title="finished", message='train finished')
            elif value == 'BiLSTM':
                global BiLSTM_Trained
                if BiLSTM_Trained == True:
                    return
                BiLSTM_Trained = True
                bilstm_train(
                    (Btrain_word_lists, Btrain_tag_lists),
                    (Bdev_word_lists, Bdev_tag_lists),
                    (Btest_word_lists, Btest_tag_lists),
                    crf_word2id, crf_tag2id
                )
                tkinter.messagebox.showinfo(title="finished", message='train finished')
            else:
                tkinter.messagebox.showinfo(title="error", message='ERROR')
        else:
            tkinter.messagebox.showinfo(title='worning', message='please select one first')

    def solve():
        global HMM_Trained
        global CRF_Trained
        global BiLSTM_Trained

        if len(lb.curselection()) == 1:
            input_data = e1.get('1.0', 'end')
            deal(input_data)
            value = lb.get(lb.curselection())
            if value == 'HMM':
                global HMM_Trained
                if HMM_Trained == False:
                    tkinter.messagebox.showinfo(title="error", message='please train first')
                    return

                dc_word_lists, dc_tag_lists = build_corpus("dc", make_vocab=False)

                hmm_model = load_model(HMM_MODEL_PATH)
                pred_tags = hmm_model.test(dc_word_lists, word2id, tag2id)

                dc_word_lists = flatten_lists(dc_word_lists)
                pred_tags = flatten_lists(pred_tags)

                Str = ''
                for i in range(0, len(dc_word_lists)):
                    Str = Str + (str(dc_word_lists[i]) + ' ' + str(pred_tags[i]) + '\n')
                e2.delete('1.0', 'end')
                e2.insert('end', Str)
            elif value == 'CRF':
                global CRF_Trained
                if CRF_Trained == False:
                    tkinter.messagebox.showinfo(title="error", message='please train first')
                    return

                dc_word_lists, dc_tag_lists = build_corpus("dc", make_vocab=False)

                crf_model = load_model(CRF_MODEL_PATH)
                pred_tags = crf_model.test(dc_word_lists)

                dc_word_lists = flatten_lists(dc_word_lists)
                pred_tags = flatten_lists(pred_tags)

                Str = ''
                for i in range(0, len(dc_word_lists)):
                    Str = Str + (str(dc_word_lists[i]) + ' ' + str(pred_tags[i]) + '\n')
                e2.delete('1.0', 'end')
                e2.insert('end', Str)
            elif value == 'BiLSTM':
                global BiLSTM_Trained
                if BiLSTM_Trained == False:
                    tkinter.messagebox.showinfo(title="error", message='please train first')
                    return

                dc_word_lists, dc_tag_lists = build_corpus("dc", make_vocab=False)
                Bdc_word_lists, Bdc_tag_lists = prepocess_data_for_lstmcrf(
                    dc_word_lists, dc_tag_lists, test=True
                )

                bilstm_model = load_model(BiLSTMCRF_MODEL_PATH)
                lstmcrf_pred, target_tag_list = bilstm_model.test(Bdc_word_lists, Bdc_tag_lists, crf_word2id,
                                                                  crf_tag2id)

                Bdc_word_lists = flatten_lists(Bdc_word_lists)
                lstmcrf_pred = flatten_lists(lstmcrf_pred)
                Str = ''
                a = 0
                b = 0
                for i in range(0, len(Bdc_word_lists)):
                    if Bdc_word_lists[a] == '<end>':
                        a = a + 1
                        continue
                    Str = Str + (str(Bdc_word_lists[a]) + ' ' + str(lstmcrf_pred[b]) + '\n')
                    b = b + 1
                    a = a + 1
                e2.delete('1.0', 'end')
                e2.insert('end', Str)
            else:
                tkinter.messagebox.showinfo(title="error", message='ERROR')
        else:
            tkinter.messagebox.showinfo(title='worning', message='please select one first')

    window = tk.Tk()
    window.title('NLP NER')
    window.geometry('700x500')
    lb = tk.Listbox(window)
    list_items = ['HMM', 'CRF', 'BiLSTM']
    for item in list_items:
        lb.insert('end', item)

    e1 = tk.Text(window, width=35, height=35)
    e2 = tk.Text(window, width=25, height=35)
    ft = tf.Font(size=10)
    e2.config(font=ft)
    e1.place(x=150, y=10, anchor='nw')
    e2.place(x=500, y=10, anchor='nw')

    b1 = tk.Button(window, text='Train Model', command=trainModel)
    b2 = tk.Button(window, text='Evaluate', command=evalModel)
    b3 = tk.Button(window, text='Solve', command=solve)

    b3.place(height=30, width=80, x=410, y=160, anchor='nw')
    b2.place(height=30, width=80, x=10, y=190, anchor='nw')
    b1.place(height=30, width=80, x=10, y=160, anchor='nw')
    lb.place(height=150, width=75, x=10, y=10, anchor='nw')

    window.mainloop()


'''
马云，男，汉族，中共党员，1964年9月10日生于浙江省杭州市，祖籍浙江省嵊州市谷来镇， 阿里巴巴集团主要创始人，现担任日本软银董事、大自然保护协会中国理事会主席兼全球董事会成员、华谊兄弟董事、生命科学突破奖基金会董事、联合国数字合作高级别小组联合主席。
1988年毕业于杭州师范学院外语系，同年担任杭州电子工业学院英文及国际贸易教师，1995年创办中国第一家互联网商业信息发布网站“中国黄页”，1998年出任中国国际电子商务中心国富通信息技术发展有限公司总经理，1999年创办阿里巴巴，并担任阿里集团CEO、董事局主席。
'''

'''
中国科学院地理研究所的前身是中国地理研究所。1937年中央研究院开始筹建中国地理研究所，并聘李四光为国立中央研究院地理研究所筹备处主任，后因战乱及经济问题未果。1940年8月由中英庚款董事会创建的中国地理研究所在重庆北碚正式成立，黄国璋任所长。所内设自然地理、人生地理、大地测量、海洋四个学科组、分别由李承三、林超、曹谟及马廷英四人主持。所内并设地图、图书资料、事务等室，全所职工约40人。1946年上半年，黄国璋辞职，由李承三代理所长。1946年8月，中英庚款董事会由于无法维持中国地理研究所，将中国地理研究所改隶属国民党教育部。因李承三离所，所长由林超继任。1947年夏中国地理研究所由重庆北碚迁至江苏南京。1948—1949年由罗开富代理所长。
'''
