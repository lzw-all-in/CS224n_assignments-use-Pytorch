# CS224n_assignments
#### 已经把第三个作业传上来了，但是q3_gru.py文件没有进行编写，由于需要使用dynamic_rnn从而实现自动padding，而Pytorch没有这个模型，虽然也可以自己手动调用Pytorch的padding函数，不过实现起来太过费力，故选择放弃。

#### 数据挖掘文件夹下的是自己学校的PJ，做的关于知乎文章的情感分析

#### 视频看的是Youtube上的视频，感觉B站上翻译的不太好。并且存在字幕和声音不同步的现象。By the way, Youtube上面多了一节期中考试的复习视频。学而时习之~~~~
## 说明
- 源码使用Python2.7编写，由于版本比较过时，所以使用的3.x版本(本地用的3.6，服务器用的3.5)，修改了其中的一些BUG。
- 自己没有采用tensorflow的框架，而是使用的Pytorch。
- **.log** 文件是代码在服务器上运行时打印输出的信息，可以了解模型的运行状态。
- CS224n 2017和2018期中考试非常有价值
- 自己的代码写的有点乱希望各位看官不要介意 %>_<%

资源名 | 资源链接
---|---
原作业数据(百度网盘) | [link](https://pan.baidu.com/s/17ripXND-xSzzP4vgppseig)
课程的默认Final PJ页面 | [link](http://web.stanford.edu/class/cs224n/default_project/index.html)
CS224n官方页面链接(含start code和solution还有各种参考资料) | [link](http://web.stanford.edu/class/cs224n/syllabus.html) 
Pytorch教程 | [link](https://github.com/chenyuntc/pytorch-book)
课程笔记(五星推荐) | [link](https://github.com/stanfordnlp/cs224n-winter17-notes)
原版本参考代码(Python2.7 & tensorflow) | [link](https://github.com/hankcs/CS224n)
最新版自然语言处理综论(Speech and Language Processing)| [link](http://web.stanford.edu/~jurafsky/slp3/)
---


# assignment one

### 说明
- 一些基础性的修改，比如print括号问题
- utils.treebank.py下的sentence和sent_lables函数做了修改，原因是解码的问题。感觉自己的解决方式有点暴力：如果遇到有解码问题的句子就跳过。如果有更好的解决方式请联系我 ：)
- 作业一的运行后的图片转移到assignment1 下面去了
- 模型最后的损失iter_ 40000: 9.386926(可在.log文件中查看)

# assignment two
### 说明
- 自己修改后的模型的文件名有个extention的后缀
- 源代码为tensorflow写的，自己用pytorch全部改写了
- utils文件下的代码，只做了一点点的修改，主要是转化数据类型，几乎没有什么变化。
- 由于随机的问题，可能结果会不太一样，但不会差太多。

### 最终结果
- .log的文件中看到原始模型 test UAS: 89.40  
- 自己修改后的模型 test UAS: 90.05
- 修改的模型：额外加入一个隐层，并将两个隐层的输出加入到最终softmax中(详情见代码 q2_parser_model_extension.py)

# assignment three
### 说明
- 模型改动很大，具体的需要看代码才能知道
- q3_gru.py文件没有实现，由于Pytorch缺少必要的库函数

### 最终结果
### q1   
**Token-level confusion matrix:**  

go\gu   |	PER     |	ORG     |	LOC    | 	MISC    |	O   
---|---|---|---|---|---|  
PER   |  	2933.00 |	73.00   |	58.00   |	10.00   |	75.00   
ORG    | 	125.00  |	1691.00 |	101.00  |	50.00   |	125.00  
LOC    | 	36.00   |	125.00  |	1872.00 |	21.00   |	40.00   
MISC   | 	45.00   |	64.00   |	52.00   |	995.00  |	112.00  
O      | 	44.00   |	63.00   |	15.00   |	26.00   |	42611.00

**Token-level scores:**  

label|	acc  |	prec |	rec  |	f1  
---|---|---|---|---|  
PER  |	0.99 |	0.92 |	0.93 |	0.93 
ORG  |	0.99 |	0.84 |	0.81 |	0.82 
LOC  |	0.99 |	0.89 |	0.89 |	0.89 
MISC |	0.99 |	0.90 |	0.78 |	0.84 
O    |	0.99 |	0.99 |	1.00 |	0.99 
micro|	0.99 |	0.98 |	0.98 |	0.98 
macro|	0.99 |	0.91 |	0.88 |	0.90 
not-O|	0.99 |	0.89 |	0.87 |	0.88 

**Entity level P/R/F1: 0.82/0.85/0.83**  

### q2  
**Token-level confusion matrix:**  
go\gu   |	PER     |	ORG     |	LOC    | 	MISC    |	O   
---|---|---|---|---|---|  
PER     |	2968.00 |	37.00   |	82.00   |	11.00   |	51.00   
ORG    | 	108.00  |	1697.00 |	115.00  |	54.00   |	118.00  
LOC     |	31.00   |	77.00   |	1945.00 |	7.00    |	34.00   
MISC    |	32.00   |	60.00   |	63.00   |	1003.00 |	110.00    
O       |	39.00   |	42.00   |	18.00   |	21.00   |	42639.00  


**Token-level scores:**  
label|	acc  |	prec |	rec  |	f1  
---|---|---|---|---|  
PER  |	0.99 |	0.93 |	0.94 |	0.94   
ORG  |	0.99 |	0.89 |	0.81 |	0.85   
LOC  |	0.99 |	0.87 |	0.93 |	0.90   
MISC |	0.99 |	0.92 |	0.79 |	0.85   
O    |	0.99 |	0.99 |	1.00 |	0.99   
micro|	0.99 |	0.98 |	0.98 |	0.98   
macro|	0.99 |	0.92 |	0.89 |	0.91   
not-O|	0.99 |	0.91 |	0.88 |	0.89   

**Entity level P/R/F1: 0.84/0.86/0.85**  

### q3


**Token-level confusion matrix:**  
go\gu   |	PER     |	ORG     |	LOC    | 	MISC    |	O   
---|---|---|---|---|---|  
PER     |	2976.00 |	37.00   |	48.00   |	13.00   |	75.00   
ORG     |	127.00  |	1721.00 |	75.00   |	86.00   |	83.00   
LOC     |	37.00   |	86.00   |	1916.00 |	23.00   |	32.00   
MISC    |	39.00   |	49.00   |	40.00   |	1034.00 |	106.00  
O       |	53.00   |	59.00   |	18.00   |	31.00   |	42598.00  



**Token-level scores:**  
label|	acc  |	prec |	rec  |	f1  
---|---|---|---|---|  
PER  |	0.99 |	0.92 |	0.95 |	0.93  
ORG  |	0.99 |	0.88 |	0.82 |	0.85  
LOC  |	0.99 |	0.91 |	0.91 |	0.91  
MISC |	0.99 |	0.87 |	0.82 |	0.84  
O    |	0.99 |	0.99 |	1.00 |	0.99  
micro|	0.99 |	0.98 |	0.98 |	0.98  
macro|	0.99 |	0.92 |	0.90 |	0.91  
not-O|	0.99 |	0.90 |	0.89 |	0.90  
**Entity level P/R/F1: 0.85/0.87/0.86**  
