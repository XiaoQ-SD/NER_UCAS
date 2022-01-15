# NER_UCAS
NER project for NLP course 2021

### 主要参考
代码主要参考自：

https://github.com/luopeixiang/named_entity_recognition

https://github.com/jiesutd/LatticeLSTM

https://github.com/buppt/ChineseNER

学习博客来自：

https://zhuanlan.zhihu.com/p/61227299

https://blog.csdn.net/omnispace/article/details/89953473

https://blog.csdn.net/cuihuijun1hao/article/details/79405740

https://blog.csdn.net/Tianweidadada/article/details/102691175

https://www.cnblogs.com/shwee/p/9427975.html

参考文献：

Lample G, Ballesteros M, Subramanian S, et al. Neural architectures for named entity recognition[J]. arXiv preprint arXiv:1603.01360, 2016.

Lafferty J, McCallum A, Pereira F C N. Conditional random fields: Probabilistic models for segmenting and labeling sequence data[J]. 2001.

### 操作方法

在实现HMM，CRF，BiLSTM三个模块后，将三个方法进行整合，利用python tkinter库制作图形界面以实现交互。

界面如图所示：

![](../hexo_blog/blog/hexo/source/images/ui.png)

左上为方法选择，即三种方法中选择所采用的方法。两个按钮分别表示训练该选中模型和评估该选中模型。中间的文本框为输入框，按钮Solve为分析操作，即对输入的文本进行命名实体识别分析，右侧文本框为结果输出框，此外，评估模型的输出结果一部分于终端中显示，一部分与输出框中显示。

界面代码详见ui.py。