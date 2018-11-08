# CS224n_assignments

## 说明

- 源码使用Python2.7编写，由于版本比较过时，所以使用的3.6版本，修改了其中的一些BUG。
- 自己没有采用tensorflow的框架，而是使用的Pytorch。自己也是刚刚开始学习这个框架。
- 官方页面链接(含start code和solution) [link](http://web.stanford.edu/class/cs224n/syllabus.html) 
- Pytorch教程链接 [link](https://github.com/chenyuntc/pytorch-book)
- 参考了github上之前有人用Python2.7写的代码 [link](https://github.com/hankcs/CS224n)
- 课程笔记(五星举荐) [link](https://github.com/stanfordnlp/cs224n-winter17-notes)
- 欢迎和我一起讨论，自己也刚刚开始接触 : )
---

# assigment one

### word2vec

![image](https://note.youdao.com/yws/api/personal/file/A4A10D2E3BC04B2D8659417E9AAF8C0C?method=download&shareKey=b0fdede89f787b2f28a17e89c0be3b41)

### sentiment
![image](https://note.youdao.com/yws/api/personal/file/9B76525F990648D78CF5672C35992152?method=download&shareKey=d83c7ac9cf477342337d25030f857470)
![image](https://note.youdao.com/yws/api/personal/file/8A6B212F21714221A723386CEBA67B21?method=download&shareKey=9ed57f92161aa7f825a3a22ebba71229)

- 感觉和官方的solution还是有点区别的

#### 修改了的函数
- 一些基础性的修改，比如print括号问题
- utils.treebank.py下的sentence和sent_lables函数做了修改，原因是解码的问题。感觉自己的解决方式有点暴力，如果遇到有解码问题的句子就跳过。如果有更好的解决方式请联系我 ：)