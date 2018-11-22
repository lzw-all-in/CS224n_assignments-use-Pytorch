# CS224n_assignments
#### 这学期课程比较多，缓慢更新中~~~~~~~~~~~~~~~^_\^
#### 最近在做课程PJ，所以后面作业更新应该会等一段时间，目前学校的课自己打算做一个对知乎文章进行情感分类。自己数据已经爬取到了，还在等同学标数据( T___T ) 。目测会尝试朴素贝叶斯还有各种RNN的各种变体。做完后会一同传上来\ ( > < ) / 
#### 视频看的是Youtube上的视频，感觉B站上翻译的不太好。By the way, Youtube上面多了一节期中考试的复习课，也很有用哦。学而时习之~~~~
#### 学习这件事情不能急要慢慢来，欲速则不达哦p( ^ O ^ )q
## 说明

- 源码使用Python2.7编写，由于版本比较过时，所以使用的3.x版本(本地用的3.6，服务器用的3.5)，修改了其中的一些BUG。
- 自己没有采用tensorflow的框架，而是使用的Pytorch。自己也是刚刚开始学习这个框架。
- assignment one的数据太大了，所以没有传。使用shell脚本就可以获得。
- **.log** 文件是代码在服务器上运行时打印输出的信息，可以了解模型的运行状态。
- CS224n 2017和2018期中考试非常有价值

资源名 | 资源链接
---|---
CS224n官方页面链接(含start code和solution还有各种参考资料) | [link](http://web.stanford.edu/class/cs224n/syllabus.html) 
Pytorch教程 | [link](https://github.com/chenyuntc/pytorch-book)
课程笔记(五星推荐) | [link](https://github.com/stanfordnlp/cs224n-winter17-notes)
原版本参考代码(Python2.7 & tensorflow) | [link](https://github.com/hankcs/CS224n)
最新版自然语言处理综论(Speech and Language Processing)| [link](http://web.stanford.edu/~jurafsky/slp3/)
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
- .log的文件中看到原始模型 test UAS: 89.40  
- 自己修改后的模型 test UAS: 90.05
- 修改的模型：额外加入一个隐层，并将两个隐层的输出加入到最终softmax中(详情见代码 q2_parser_model_extension.py)
