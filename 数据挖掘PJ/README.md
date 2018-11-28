# 知乎文本情感分析 
**说明**
- 朴素贝叶斯伪代码参考文档 [http://web.stanford.edu/~jurafsky/slp3/4.pdf](http://web.stanford.edu/~jurafsky/slp3/4.pdf)
- PJ所有数据链接 [https://pan.baidu.com/s/1XmlBDasLaIiQO04dMr2nPA](https://pan.baidu.com/s/1XmlBDasLaIiQO04dMr2nPA)
- 从知乎上爬取了一些用户信息，还有一些文章，在课程的presetation上做了一些展示，感觉用户信息的数据分析网上已经很详细了，下面会贴一些网上没有的图
- 上传文件中没有关于数据分析的文件，只是贴了2张我们觉得有意思的图
- articles.csv里面存放的1000篇文章，target.csv里面存放我们人工打的标签
- 我们把文章做了一下过滤200字一下的丢掉了，还剩916篇
- 目前人工打了90个标签(原谅我们组只有一个人负责标数据(∩＿∩)) 
- 目前属于占坑吧，会慢慢把这个文件填满的(\*\^.\^\*) ~~~~持续更新中
---

> 爬取了某些板块下的1000篇文章用来做词频统计

### 上交板块和复旦板块某些关键词词频对比
![image](https://note.youdao.com/yws/api/personal/file/D2BBBA1E43F84F258CE44B0F8C24C997?method=download&shareKey=387e4f30fafb87697623426e8cbceaf2)
**惊奇发现我旦保研和出国的词频比上交要少(泣)**  

### 将词频从高到低排列，对比不同板块下高词频的词的重合度
![image](https://note.youdao.com/yws/api/personal/file/D1C089A948974ABCA49708879DAC2C85?method=download&shareKey=2b0bab8376030bb4ebafc4af57194216)

----
> 模型数据，目前只使用了朴素贝叶斯，后面会尝试各种变体的RNN
### Naive Bayes混淆矩阵(90条数据)
![image](https://note.youdao.com/yws/api/personal/file/33E5C5DBF8DB4C659E55171045C34FC7?method=download&shareKey=167e559f7814a0cd46563b5de133499f)

### Naive Bayes ROC曲线(90条数据)
![image](https://note.youdao.com/yws/api/personal/file/52F0784BD7DB499C96E4349C00063800?method=download&shareKey=70af34ba9f39988e7cf12db14fe7bcfc)

