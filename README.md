# TCN_TRAS_power
use TCN and Transformer model for "Hourly Energy Consumption" data
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

## References
 <https://www.twblogs.net/a/5d6dc709bd9eee541c33c0b2>  
 <https://meetonfriday.com/posts/7c0020de/>  
 <https://medium.com/@a5560648/dropout-5fb2105dbf7c>
