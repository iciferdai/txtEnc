
# txtEnc
> 手搓轻量级文本编码器 txtEnc，实践MLM预训练、语义相似度微调与向量数据库检索应用  
> Build a lightweight text encoder txtEnc from scratch, and practice MLM pre-training, semantic similarity fine-tuning, and retrieval with vector databases

## 1. 项目介绍 / Introduce

以 BERT 为代表的编码器架构模型，是 Transformer 在 Encoder 技术分支上的重要发展方向，如今在语义 Embedding、检索与召回等领域发挥着关键作用。

本项目基于 PyTorch 从零搭建轻量级文本编码器 txtEnc（参考前置项目：TransDemo），不直接调用框架封装好的预训练模型接口，聚焦理解 Transformer 编码器结构、掩码语言模型（MLM）、自注意力与语义表征等核心机制。

本项目先通过 MLM 掩码填词任务完成模型预训练，再通过微调实现相似句子向量逼近与语义相似度计算，覆盖模型构建、数据处理、预训练、微调与推理全流程，实现理论与实战结合，深入掌握编码器类模型的底层原理与语义表示能力。


## 2.  效果 / Effect Demo

#### 使用chroma，召回Top 3：
```log
Press send input: 中东冲突升级，伊朗关闭霍尔木兹海峡，全球石油运输通道受阻，国际油价大幅跳涨。
======= Search Results =======
id: 11, distances: 0.0037093758583068848, message: {'txt': '中东地缘冲突持续升级，伊朗方面宣布关闭霍尔木兹海峡，全球约两成石油运输通道受阻，国际油价大幅走高，避险资产价格同步上行，全球主要股指出现明显波动。'}
id: 82, distances: 0.003950238227844238, message: {'txt': '中东地缘冲突升级，伊朗宣布关闭霍尔木兹海峡，全球约两成石油运输通道受到影响。'}
id: 117, distances: 0.004095911979675293, message: {'txt': '国际保险机构将波斯湾、红海相关海域列入战争险高风险区域，船舶保险费率大幅上调，全球航运成本明显上升。'}
```

#### 使用Qdrant，召回Top 3：

```log
Press send input: A股权益类ETF两日净流入近380亿元，交易活跃度明显提升。
======= Search Results =======
id: 26, vector: None, payload: {'text': '3月首周A股权益类ETF两日净流入近380亿元，资金借道ETF布局市场，交易活跃度明显提升。'}, score: 0.9934652930896897
id: 67, vector: None, payload: {'text': '长安汽车发布股份回购计划，拟使用10亿至20亿元自有资金回购A股与B股股份，用于减少注册资本。'}, score: 0.9931031273800731
id: 92, vector: None, payload: {'text': '沪深北交易所优化再融资安排，放宽破发企业竞价定增限制，支持未盈利科技企业融资。'}, score: 0.9929637021856649
```

#### MLM Mask验证：

```log
====================== Target <-> Predicted ======================
<CLS>欧洲2月制造业PMI低于荣枯线，制造业复苏乏力，欧元区经济增长压力显现。
<CLS>欧洲2月制造业PMI低于荣枯线，制造业复苏乏力，欧元区经济增长压力显现。
======================= Mask <-> Predicted =======================
<CLS>欧洲2月制<MASK><MASK>PMI低于荣枯线，制造<MASK><MASK>苏乏力，欧元区经济增长压力显现。
------造业-----------业复----------------
==================================================================
...(略)...
====================== Target <-> Predicted ======================
<CLS>2月A股两融新开户数同比增长，融资余额与日均融资买入额较节前明显回升，杠杆资金活跃度提升。
<CLS>2月A股两融新开户数同比器长，融资余额与日均融资买入额较节前明显回升，杠杆资金活跃度提升。
======================= Mask <-> Predicted =======================
<CLS>2月A股两<MASK>新开户数<MASK>比器长，融资余额与日均<MASK>资买入额较节前明显回<MASK>，杠杆<MASK>金<MASK>跃度提升。
------融----同-----------融----------升---资-活-----
==================================================================
```


## 3. 训练 / Training

#### 训练相关信息：
- 使用120条近期热点新闻句子，并人工标记1(正向)、0(中立)、-1(负向)
- Batch Size设置为8，MAX_LEN=100
- 预训练采用Adam优化器：lr=1e-4
- 微调训练采用AdamW优化器：lr=2e-5, weight_decay=0.01
- 预训练使用交叉熵损失：nn.CrossEntropyLoss
- 微调训练使用自定义contrast_loss：输入锚点anchor（训练样本）、正样本positive、负样本negative，比例是1:1:8（基础设置中：NEGATIVE_SAMPLE_NUM = 8），计算锚点与正负样本的余弦相似度（cosine_similarity），最后得到loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum)).mean()
- 预训练进行1000个epoch，微调则训练100个epoch

#### Loss Dashboard（预训练）：
![Loss_Dashboard](https://github.com/iciferdai/txtEnc/blob/main/pictures/Loss_Dashboard.png)

#### Loss Dashboard（微调训练）：
![Loss_Dashboard_SFT](https://github.com/iciferdai/txtEnc/blob/main/pictures/Loss_Dashboard_SFT.png)


## 4. 实现介绍 / Implementation Introduction
>  前置项目为：TransDemo，本项目介绍以差异为主，具体实现请结合参考前置项目以及代码

#### 1.  模型：
- 主体基于3层Transformer Encoder Layer
- 采用标准固定位置编码，POS_ENCODING_BASE = 10000.0
- FFN的激活函数由ReLU升级为GELU

#### 2.  预训练：
- 输入的文本序列以标签[CLS_ID]开头，作为本条文本的向量标签（可替换为其他取标签方式，如mean，sum等）
- 对输入序列进行MLM Mask：随机选中15%的token（可掩码位置），对其中的80%替换为MASK_ID（掩码）以及10%替换为词表中的随机词（混淆）
- 预训练任务是对掩码的位置，预测其原始值

#### 3.  微调训练：
- 冻结前两层Encoder以及Embedding，解冻最后一层Encoder
- 对已经标记的训练样本（1-正向、0-中立、-1-负向）逐条遍历，每条锚点样本随机取1条同类样本做正样本，8条不同类样本做负样本
- 对锚点样本、正样本、负样本，先分别进行模型推理，得到各自的CLS向量，然后将各自的向量，按前述contrast_loss计算loss，做反向传播和优化

#### 4.  推理/向量检索：
- 预训练后推理，直接对原数据遍历推理，取最大概率预测值（贪心），查看验证结果
- 微调后推理，对原数据逐条做模型推理，取CLS向量作为其向量，然后与文本匹配后全部存入向量数据库中，验证采用Chroma和Qdrant两种向量数据库，均采用内存模式做简单验证，不做持久化存储；对用户输入，做模型推理，得到CLS向量，按向量相似度，从向量数据库中召回Top N的向量及其原始文本描述


## 5. 过程总结 / Conclusion

- 微调时使用更低的学习率
- 不同向量数据库的差异（如相似度 or 距离）

---
##  结束 / End
>:loudspeaker: Notice：  
>本项目为个人学习与实验性项目  
> This is personal learning and experimental project
