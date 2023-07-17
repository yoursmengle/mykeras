# 项目说明：
1. keras是一个基于python的库，用于建立一个深度神经网络；
2. keras除了依赖python以外，还依赖其它的一些库，主要包括numpy,theano,PIL等，需要额外安装，比如在CENT OS下可使用yum install numpy或者 pip install numpy等方式；
3. mnist目录,包含了42000张手写数字的图片，格式是单色28X28,用于本例子训练
4. dive_into_keras/cnn.py 训练程序，执行时会首先建立一个神经网络，然后调入上面的mnist目录中的图片对网络进行训练，并将多轮训练中错误率最低的模型保存为model.pkl文件。
5. jhzhou 目录是个人对上述model.pkl文件应用的一个小例子，可对命令行输入的单个图片进行识别。
