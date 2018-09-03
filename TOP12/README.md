## TSA2018-rank12
#### 环境配置：
512G内存，12G显存以上显卡
#### 依赖库：
python3.5版本，需要最新版的sklearn，numpy，tqdm，pandas，tensorflow 1.4.0
#### 原始数据输入：
input_round1：存放初赛数据;input_round2：存放复赛数据
#### 特征使用：
除部分原始特征外，加了三个人工特征：uid_cnt:uid的计数; aid_cnt：aid计数; uid_cvr：uid的转化率
#### 模型：
主要为deepffm，在@nzc的开源代码上稍微修改，参数见src/main.py
#### 代码运行：
依次执行src目录下的preprocess.py与main.py

