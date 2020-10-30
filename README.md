# centernet-inst
A little experiment combining [Centernet](https://arxiv.org/abs/1904.07850) and [SOLOv2](https://arxiv.org/abs/2003.10152).


# how
Add two heads to Centernet to predict the convolution kernel parameters and the segmentation feature map. [here](https://github.com/gakkiri/centernet-inst/blob/master/centernet/centernet_head.py#L42)

# training curve
![total](https://github.com/gakkiri/centernet-inst/blob/master/imgs/total_loss.png?x-oss-Process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)
![mask](https://github.com/gakkiri/centernet-inst/blob/master/imgs/mask_loss.png?x-oss-Process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)  


