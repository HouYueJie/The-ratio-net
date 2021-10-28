# 数据说明
- xy_text_cnn_train   SST2降维数据  

# 运行说明
  传统网络：python main.py MLP or python main.py RBF  
  新网络：python main.py Pade      

# 结果说明
  会在ckpt_** 中产生 model_weights  
  会在logging 中产生 模型训练日志  
  在retsult_plot 中运行 python plot.py test --> 依据日志画出test acc曲线图  

# 配置网络结构说明
在config.py文件中，配置 struc  
在utils.py文件中，配置是否需要z-score标准化或Max_min归一化  
