1.主数据集2w6数据量，划分为70%作为训练集,15%作为验证集,15%作为测试集。跨细胞数据集1w2数据量作为独立测试集测试模型性能。正负样本1:1。正样本来源于encode项目bed实验数据，负样本为真实非结合区域随机采样构造。

2.编码方式：one-hot

3.模型架构cnn+bilstm+attention+全连接层

4.输出量化指标：Accuracy	Precision	Recall	F1-Score	AUC-ROC	AUPRC
