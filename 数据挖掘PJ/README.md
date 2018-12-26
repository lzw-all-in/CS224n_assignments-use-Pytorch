# 知乎文本情感分析 
**说明**
- 朴素贝叶斯伪代码参考文档 [http://web.stanford.edu/~jurafsky/slp3/4.pdf](http://web.stanford.edu/~jurafsky/slp3/4.pdf)
- PJ所有数据链接 [https://pan.baidu.com/s/1qnyGIZUhLQpjZvKr3rR3Kg](https://pan.baidu.com/s/1qnyGIZUhLQpjZvKr3rR3Kg)
- 从知乎上爬取了一些用户信息，还有一些文章，在课程的presetation上做了一些展示，感觉用户信息的数据分析网上已经很详细了，下面会贴一些网上没有的图
- 上传文件中没有关于数据分析的文件，只是贴了2张我们觉得有意思的图
- articles.csv里面存放的1000篇文章，target.csv里面存放我们人工打的标签
- 最后由于时间关系只人工标注了400+篇

---

> 爬取了某些板块下的1000篇文章用来做词频统计

### 上交板块和复旦板块某些关键词词频对比
![image](https://note.youdao.com/yws/api/personal/file/D2BBBA1E43F84F258CE44B0F8C24C997?method=download&shareKey=387e4f30fafb87697623426e8cbceaf2)
**惊奇发现我旦保研和出国的词频比上交要少(泣)**  

### 将词频从高到低排列，对比不同板块下高词频的词的重合度
![image](https://note.youdao.com/yws/api/personal/file/D1C089A948974ABCA49708879DAC2C85?method=download&shareKey=2b0bab8376030bb4ebafc4af57194216)

----
> 模型数据
### ACC(400+条数据)
![image](https://note.youdao.com/yws/api/personal/file/A9DA21CAE3774324997330CB081E8E74?method=download&shareKey=9eee11a71d1f534334243ce73d2f5125)

### ROC曲线(400+条数据)
![image](https://note.youdao.com/yws/api/personal/file/45CC7BEEBE064A84BA45327AB32BCCB2?method=download&shareKey=86ec69792a93ae464b2c887dca3e3a14)

### LSTM，GRU，RNN训练过程中的损失值
![image](https://note.youdao.com/yws/api/personal/file/F5FDE0CA9DCC4EC390603E8E0CE1BD9B?method=download&shareKey=a606aca65df0dca34ebf9a4589139132)
