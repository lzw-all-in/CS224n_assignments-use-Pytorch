# CS224n_assignments
#### 这学期课程比较多，缓慢更新中~~~~~~~~~~~~~~~^_\^
#### 今天已经把PJ的初稿传上来了，趁着学习CS224n自己选了一个情感分析的题目。详情请见数据挖掘PJ文件说明
#### 最近在做课程PJ，所以后面作业更新应该会等一段时间，目前学校的课自己打算做一个对知乎文章进行情感分类。自己数据已经爬取到了，还在等同学标数据( T___T ) 。
#### 视频看的是Youtube上的视频，感觉B站上翻译的不太好。并且存在字幕和声音不同步的现象。By the way, Youtube上面多了一节期中考试的复习课，也很有用哦。学而时习之~~~~
## 说明
- 抱歉哈，之前一直不知道作业一的代码无法下载，原作业数据的百度网盘已经贴在下面了
- 源码使用Python2.7编写，由于版本比较过时，所以使用的3.x版本(本地用的3.6，服务器用的3.5)，修改了其中的一些BUG。
- 自己没有采用tensorflow的框架，而是使用的Pytorch。自己也是刚刚开始学习这个框架。
- 前端hijack
- **.log** 文件是代码在服务器上运行时打印输出的信息，可以了解模型的运行状态。
- CS224n 2017和2018期中考试非常有价值
- 自己的代码写的有点乱希望各位看官不要介意 %>_<%

资源名 | 资源链接
---|---
原作业数据(百度网盘) | [link](https://pan.baidu.com/s/1IXZXzHpm1MO19hl6TwSkcQ)
CS224n官方页面链接(含start code和solution还有各种参考资料) | [link](http://web.stanford.edu/class/cs224n/syllabus.html) 
Pytorch教程 | [link](https://github.com/chenyuntc/pytorch-book)
课程笔记(五星推荐) | [link](https://github.com/stanfordnlp/cs224n-winter17-notes)
原版本参考代码(Python2.7 & tensorflow) | [link](https://github.com/hankcs/CS224n)
最新版自然语言处理综论(Speech and Language Processing)| [link](http://web.stanford.edu/~jurafsky/slp3/)
---


# assignment one

#### 说明
- 一些基础性的修改，比如print括号问题
- utils.treebank.py下的sentence和sent_lables函数做了修改，原因是解码的问题。感觉自己的解决方式有点暴力：如果遇到有解码问题的句子就跳过。如果有更好的解决方式请联系我 ：)
- 作业一的运行后的图片转移到assignment1 下面去了
- 模型最后的损失iter_ 40000: 9.386926(可在.log文件中查看)

# assignment two
#### 说明
- 自己修改后的模型的文件名有个extention的后缀
- 源代码为tensorflow写的，自己用pytorch全部改写了
- utils文件下的代码，只做了一点点的修改，主要是转化数据类型，几乎没有什么变化。
- 由于随机的问题，可能结果会不太一样，但不会差太多。

### 最终结果
- .log的文件中看到原始模型 test UAS: 89.40  
- 自己修改后的模型 test UAS: 90.05
- 修改的模型：额外加入一个隐层，并将两个隐层的输出加入到最终softmax中(详情见代码 q2_parser_model_extension.py)
