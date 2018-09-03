## 环境要求
    1、在ti-one tensorflow(python3.5) 组件上运行通过， 代码里使用pip安装了pickle。
## 步骤说明
    1、将原始数据merge并分割为：训练文件、测试文件、以及预测文件，按列存储到文件夹中。
    2、将初赛的原始数据merge，分割为：训练文件、预测文件，按列存储到文件夹中。
    3、增加特征，依此为用户计数、用户与广告特征的交叉计数、分aid的单值特征的信息熵、多值特征的信息熵。
## 结构说明
- data:存放数据文件以及生成的特征文件
    - train、test、predict 为复赛数据的特征文件。
    - train1、predict1 为初赛数据的特征文件
    - userFeature.data、userFeature.csv、train.csv、test1.csv、test2.csv 为复赛数据
    - userFeature2.data、userFeature2.csv、train1.csv、test11.csv、test12.csv 为初赛数据
- models：模型目录
    - nffm.py 核心模型文件。
    - utils.py 打印日志的工具
- src: 源文件
    - base_data.py 处理复赛数据
    - base_data_1.py 处理初赛数据
    - add_feature_with_new.py 增加特征
    - main.py 主程序入口


#### 备注：项目由于是在ti-one上测试的，所以有一部分日志输出以及缓存文件转移的代码。如果切换环境，可能不兼容。