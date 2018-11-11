# CS224n_assignments

## 说明

- 源码使用Python2.7编写，由于版本比较过时，所以使用的3.x版本(本地用的3.6，服务器用的3.5)，修改了其中的一些BUG。
- 自己没有采用tensorflow的框架，而是使用的Pytorch。自己也是刚刚开始学习这个框架。
- assignment one的数据太大了，所以没有传。使用shell脚本就可以获得。
- 每个作业里面的.log文件是代码在服务器上运行时打印输出的信息，可以了解模型的运行状态。
- 官方页面链接(含start code和solution) [link](http://web.stanford.edu/class/cs224n/syllabus.html) 
- Pytorch教程链接 [link](https://github.com/chenyuntc/pytorch-book)
- 参考了github上之前有人用Python2.7写的代码 [link](https://github.com/hankcs/CS224n)
- 课程笔记(五星推荐) [link](https://github.com/stanfordnlp/cs224n-winter17-notes)
- 欢迎和我一起讨论，自己也刚刚开始接触 : )
---

# assigment one

#### 说明
- 一些基础性的修改，比如print括号问题
- utils.treebank.py下的sentence和sent_lables函数做了修改，原因是解码的问题。感觉自己的解决方式有点暴力：如果遇到有解码问题的句子就跳过。如果有更好的解决方式请联系我 ：)
- 自己的代码写的有点乱希望各位看官不要介意 %>_<%
### word2vec

![image](https://note.youdao.com/yws/api/personal/file/A4A10D2E3BC04B2D8659417E9AAF8C0C?method=download&shareKey=b0fdede89f787b2f28a17e89c0be3b41)

### sentiment
![image](https://note.youdao.com/yws/api/personal/file/9B76525F990648D78CF5672C35992152?method=download&shareKey=d83c7ac9cf477342337d25030f857470)
![image](https://note.youdao.com/yws/api/personal/file/8A6B212F21714221A723386CEBA67B21?method=download&shareKey=9ed57f92161aa7f825a3a22ebba71229)

# assigment two
#### 说明
- 自己修改后的模型的文件名有个extention的后缀
- 源代码为tensorflow写的，自己用pytorch全部改写了
- utils文件下的代码，只做了一点点的修改，主要是转化数据类型，几乎没有什么变化。
- 由于随机的问题，可能结果会不太一样，但不会差太多。

### 最终结果
- .log的文件中看到原始模型 test UAS: 87.40  
- 自己修改后的模型 test UAS: 88.95
- 修改的模型：额外加入一个隐层，并将两个隐层的输出加入到最终softmax中(详情见代码 q2_parser_model_extension.py)
