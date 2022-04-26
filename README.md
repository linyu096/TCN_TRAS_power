# TCN_TRAS_power
use TCN and Transformer model for forecasting on "Hourly Energy Consumption" data
## 1. Temporal Convolutional Networks
* Advantages of TCN over RNN
  - [x] Parallelism  
    `TCN can process data in parallel without requiring sequential processing like RNN.`
  - [x] Flexible receptive field  
    `The size of the receptive field of TCN is determined by the number of layers, the size of the convolution kernel, and the expansion coefficient. It can be flexibly customized according to different characteristics of different tasks.`
  - [x] Stable gradient  
    `RNNs often have problems of gradient disappearance and gradient explosion, which are mainly caused by shared parameters in different time periods. Like traditional convolutional neural networks, TCNs are less prone to gradient disappearance and explosion problems.`
  - [x] The memory is lower  
    `When RNN is used, it needs to save the information of each step, which will occupy a lot of memory. The convolution kernel of TCN is shared in one layer, and the memory usage is lower.`  
    
 ex : kernel size=2,dilations=[1,2,4,8]
### step 1 : Causal Convolutions 
![Causal Convolutions]( https://pic1.xuehuaimg.com/proxy/csdn/https://img-blog.csdnimg.cn/2019082909091041.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xlb25fd2ludGVy,size_16,color_FFFFFF,t_70 "Causal Convolutions")
### step 2 : Dilated + Causal Convolutions
In order to effectively deal with long historical information

![Dilated Causal Convolutions](https://pic1.xuehuaimg.com/proxy/csdn/https://img-blog.csdnimg.cn/20190829091941330.gif)
#### Supplementary information : Dilated Non-Causal Convolutions
(kernel size=3)
![Dilated Non-Causal Convolutions](https://pic1.xuehuaimg.com/proxy/csdn/https://img-blog.csdnimg.cn/20190829092541148.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xlb25fd2ludGVy,size_16,color_FFFFFF,t_70)
### step 3 : Add Residual block
Adaptive model depth, alleviating the phenomenon of gradient vanishing and exploding  
![Add Residual block](https://pic1.xuehuaimg.com/proxy/csdn/https://img-blog.csdnimg.cn/20190829101302335.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xlb25fd2ludGVy,size_16,color_FFFFFF,t_70)
### Supplementary information : Dropout
Prevent units from co-adapting too much, and turn on all neurons during Inference, so that the average value of each small neural network can be easily estimated, using dropout can greatly reduce the possibility of overfitting  
![Dropout](https://miro.medium.com/max/1400/1*yk0Eo4uYIk84Gu_INULcpg.png)

## Transformer model
* The motivation of Transformer is to solve the following problems of RNNs:
  * Sequential computation is difficult to parallelize.
  * Use the attention mechanism to solve the problem that RNNs cannot rely on long distances. The farther the words are, the harder it is to find each other's information.
  * Excessive RNN architecture may cause the gradient to disappear  
### Structure
Transformer consists of Encoder and Decoder, which are very different in nature and should be understood separately. The former is used to compress sequence information, and the latter is used to convert (decompress) the information extracted by the former into information required by the task.  
![Structure](https://miro.medium.com/max/746/1*6UnhXuD0hFzt7YBu1ONrcQ.png)
### Component features 1 : Self-attention
The role of Query and Key is to determine the weight of Value, and Key and Value form a token. When Query and Key are strongly related to the task, the Value corresponding to Key will be enlarged.  

![Self-attention](https://miro.medium.com/max/1400/1*fEzNWgidGdKx4Xez4C0_TQ.png)

### Component features 2 : Muti-Head Attention
The advantage of this is that each head (q, k, v) can focus on different information, some focus on local, some focus on global information, etc.  
![ Muti-Head Attention](https://miro.medium.com/max/1400/1*1AQLecxGvtjoKWBxLWj7PQ.png)

### Component features 3 : Add & Norm layer
Since the input and output have the same shape, we can use residual connection to add the output to the input. And use LayerNorm to normalize in the direction of each instance.  

![Add & Norm layer](https://miro.medium.com/max/1116/1*pTvOxquqbWusuu56UvHUhA.png)

### Component features 4 : Layer Normalization vs Batch Normalization
Layer Normalization (LN) is often used in RNN. The concept is similar to BN. The difference is that LN normalizes each sample.  
![Layer Normalization vs Batch Normalization](https://miro.medium.com/max/1400/0*oS4S5ffAoCd3qP6B.png)
### Component features 5 :Sequence Mask
The Decoder is not allowed to see future messages, so we have to cover up the vectors after i+1.  

![Sequence Mask](https://miro.medium.com/max/1116/1*pTvOxquqbWusuu56UvHUhA.png)

## References
 <https://www.twblogs.net/a/5d6dc709bd9eee541c33c0b2>  
 <https://meetonfriday.com/posts/7c0020de/>  
 <https://medium.com/@a5560648/dropout-5fb2105dbf7c>  
 <https://zhuanlan.zhihu.com/p/52477665>
 <https://ithelp.ithome.com.tw/articles/10206317>
 <https://medium.com/%E5%B7%A5%E4%BA%BA%E6%99%BA%E6%85%A7/review-attention-is-all-you-need-62a1c93c48a5>
 <https://medium.com/ching-i/transformer-attention-is-all-you-need-c7967f38af14>
