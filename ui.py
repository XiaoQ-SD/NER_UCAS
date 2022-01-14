import tkinter
import tkinter.messagebox
import tkinter.font as tf
import tkinter as tk
from data import build_corpus
from evaluate import hmm_train, hmm_eval, HMM_MODEL_PATH, crf_train, crf_eval, bilstm_train, bilstm_eval
from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from evaluating import Metrics

HMM_Trained = False
CRF_Trained = False
BiLSTM_Trained = False

train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)
crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
Btrain_word_lists, Btrain_tag_lists = prepocess_data_for_lstmcrf(train_word_lists, train_tag_lists)
Bdev_word_lists, Bdev_tag_lists = prepocess_data_for_lstmcrf(dev_word_lists, dev_tag_lists)
Btest_word_lists, Btest_tag_lists = prepocess_data_for_lstmcrf(test_word_lists, test_tag_lists, test=True)


def display():
    def showeval():
        file_object = open("cache/eval.txt")
        file_context = file_object.read()
        e2.delete('1.0', 'end')
        e2.insert('end', file_context)

    def evalModel():
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

    window = tk.Tk()
    window.title('NLP NER')
    window.geometry('800x450')
    lb = tk.Listbox(window)
    list_items = ['HMM', 'CRF', 'BiLSTM']
    for item in list_items:
        lb.insert('end', item)

    e1 = tk.Text(window, width=25, height=20, font=(14))
    e2 = tk.Text(window, width=25, height=20, font=(10))
    e1.place(x=150, y=10, anchor='nw')
    e2.place(x=500, y=10, anchor='nw')

    b1 = tk.Button(window, text='Train Model', command=trainModel)
    b2 = tk.Button(window, text='Evalute', command=evalModel)
    b3 = tk.Button(window, text='Solve')

    b3.place(height=30, width=80, x=410, y=160, anchor='nw')
    b2.place(height=30, width=80, x=10, y=190, anchor='nw')
    b1.place(height=30, width=80, x=10, y=160, anchor='nw')
    lb.place(height=150, width=75, x=10, y=10, anchor='nw')

    # var1 = tk.StringVar()
    #
    # def print_selection():
    #     value = lb.get(lb.curselection())
    #     var1.set(value)
    #
    # b1 = tk.Button(window, text='print selection', width=15, command=print_selection)
    # b1.pack()

    '''

    var = tk.StringVar()
    l = tk.Label(window, textvariable=var, bg='grey', font=(12), width=15, height=2)
    # l = tk.Label(window, text='Chinese NER', bg='grey', font=(12), width=15, height=2)
    l.pack()


    e1 = tk.Entry(window, show=None, font=(14))
    e1.pack()

    def ins_point():
        var = e1.get()
        t.insert('insert', var)

    def ins_end():
        var = e1.get()
        t.insert('end', var)

    b1 = tk.Button(window, text='insert point', command=ins_point)
    b2 = tk.Button(window, text='insert end', command=ins_end)
    b1.pack()
    b2.pack()
    t = tk.Text(window, height=3)
    t.pack()


    var1 = tk.StringVar()
    l = tk.Label

    def HIT():
        global hit
        if hit == False:
            hit = True
            var.set('hit')
        else:
            hit = False
            var.set('')

    b = tk.Button(window, text='hit', font=(12), width=10, height=1, command=HIT)
    b.pack()
    '''

    window.mainloop()
