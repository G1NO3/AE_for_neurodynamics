代码抄自thuml/Time-Series-Library  
基本思想是将时间序列做傅里叶变换，并依据主频率（本代码中为4）将整个时间序列切割并进行重新拼接成为图，不同的频率对应不同的channel（B,T,C->B,F,P,C）  
backbone选取Inception，将每一个神经元对应的四维张量喂进Encoder降到20维再重构回去，训练得到一个autoencoder  
中间的降维向量其实设计的不太好，是Bx20Dx4C，意即一个神经元分解成了4个channel和20个模  
后续要做的工作是直接对时间进行降维，**不要**把神经元拆开  
