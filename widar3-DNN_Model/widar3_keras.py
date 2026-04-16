from __future__ import print_function

import os, sys  # 用于与操作系统交互和系统相关的功能。
import numpy as np  # 用于进行科学计算和数组操作。
import scipy.io as scio  # 用于读取和写入数据文件。
import tensorflow as tf  # 谷歌开源的机器学习框架 TensorFlow。
import keras  # 一个基于 TensorFlow 的高级神经网络 API。
from keras.layers import Input, GRU, Dense, Flatten, Dropout, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, \
    TimeDistributed  # Keras 中用于构建神经网络的各种层。
from keras.models import Model, load_model  # Keras 中用于定义模型和加载模型的函数。
import keras.backend as K  # Keras 的后端接口，用于直接访问 TensorFlow 的底层功能。
from sklearn.metrics import confusion_matrix  # 用于计算混淆矩阵的函数。
# from keras.backend.tensorflow_backend import set_session # 用于设置 TensorFlow 的会话。
from sklearn.model_selection import train_test_split  # 用于划分训练集和测试集的函数。

# Parameters
use_existing_model = False  # 是否使用已存在的模型。
fraction_for_test = 0.1  # 测试集所占比例。
data_dir = 'E:/widar3/BVP/BVP/BVP/20181109-VS/6-link/user1'  # 数据存储的目录。
ALL_MOTION = [1, 2, 3, 4, 5, 6]  # 所有动作的类别列表。
N_MOTION = len(ALL_MOTION)  # 动作类别数量。
T_MAX = 0  # 时间步的最大值。
n_epochs = 30  # 训练的轮数。
f_dropout_ratio = 0.5  # Dropout 层的比例。
n_gru_hidden_units = 128  # GRU 层的隐藏单元数量。
n_batch_size = 32  # 批量训练的样本数量。
f_learning_rate = 0.001  # 学习率。


# 对输入的数据进行归一化处理
def normalize_data(data_1):
    # data(ndarray)=>data_norm(ndarray): [20,20,T]=>[20,20,T]
    data_1_max = np.concatenate((data_1.max(axis=0), data_1.max(axis=1)), axis=0).max(axis=0)
    data_1_min = np.concatenate((data_1.min(axis=0), data_1.min(axis=1)), axis=0).min(axis=0)
    if (len(np.where((data_1_max - data_1_min) == 0)[0]) > 0):
        return data_1
    data_1_max_rep = np.tile(data_1_max, (data_1.shape[0], data_1.shape[1], 1))
    data_1_min_rep = np.tile(data_1_min, (data_1.shape[0], data_1.shape[1], 1))
    data_1_norm = (data_1 - data_1_min_rep) / (data_1_max_rep - data_1_min_rep)  # 对数据进行归一化处理，使用最大值和最小值进行归一化操作。
    return data_1_norm


# 对输入的数据进行零填充处理，将数据序列的时间维度填充到指定的最大长度 T_MAX
# 零填充可以保持数据的形状一致，使得不同长度的数据序列可以输入到模型中进行训练。
def zero_padding(data, T_MAX):
    # data(list)=>data_pad(ndarray): [20,20,T1/T2/...]=>[20,20,T_MAX]
    data_pad = []
    for i in range(len(data)):
        t = np.array(data[i]).shape[2]
        data_pad.append(np.pad(data[i], ((0, 0), (0, 0), (T_MAX - t, 0)), 'constant', constant_values=0).tolist())
    return np.array(data_pad)


# 对输入的标签数据进行独热编码（One-Hot Encoding）处理，将原始的类别标签转换为对应的二进制编码表示，
# 方便在神经网络训练中作为输出标签使用。独热编码通常用于处理多分类问题。
def onehot_encoding(label, num_class):
    # label(list)=>_label(ndarray): [N,]=>[N,num_class]
    label = np.array(label).astype('int32')
    # assert (np.arange(0,np.unique(label).size)==np.unique(label)).prod()    # Check label from 0 to N
    label = np.squeeze(label)
    _label = np.eye(num_class)[label - 1]  # from label to onehot
    return _label


# 从指定路径加载数据并进行预处理
def load_data(path_to_data, motion_sel):
    global T_MAX
    data = []
    label = []  # 用于存储加载的数据和标签。
    # print(path_to_data)
    for data_root, data_dirs, data_files in os.walk(path_to_data):
        # print(data_root, data_dirs, data_files)
        for data_file_name in data_files:

            file_path = os.path.join(data_root, data_file_name)  # 构建文件的完整路径。

            try:
                data_1 = scio.loadmat(file_path)['velocity_spectrum_ro']  # 从 MAT 文件中加载名为 'velocity_spectrum_ro' 的数据。

                label_1 = int(data_file_name.split('-')[1])

                location = int(data_file_name.split('-')[2])
                orientation = int(data_file_name.split('-')[3])
                repetition = int(data_file_name.split('-')[4])

                # Select Motion 根据 motion_sel 中选定的动作进行筛选。
                if (label_1 not in motion_sel):
                    print("label_1:" + label_1)
                    continue

                # Select Location
                # if (location not in [1,2,3,5]):
                #     continue

                # Select Orientation
                # if (orientation not in [1,2,4,5]):
                #     continue

                # Normalization 对数据进行归一化处理并更新 T_MAX 的值。
                data_normed_1 = normalize_data(data_1)

                # Update T_MAX
                if T_MAX < np.array(data_1).shape[2]:
                    T_MAX = np.array(data_1).shape[2]
            except Exception:
                continue

            # Save List 将处理后的数据和标签添加到 data 和 label 列表中。
            data.append(data_normed_1.tolist())
            # print(data_normed_1.tolist())
            label.append(label_1)

    # Zero-padding 对数据进行零填充，使所有数据序列的时间维度对齐到 T_MAX。
    data = zero_padding(data, T_MAX)
    # print(data)
    # # Swap axes 交换数据的维度顺序，将数据转换为 [N, T_MAX, 20, 20] 的形式。
    data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)  # [N,20,20',T_MAX]=>[N,T_MAX,20,20']
    data = np.expand_dims(data,
                          axis=-1)  # [N,T_MAX,20,20]=>[N,T_MAX,20,20,1] 在数据的最后一个维度上增加一个维度，将数据转换为 [N, T_MAX, 20, 20, 1] 的形式。

    # Convert label to ndarray 将标签转换为 NumPy 的 ndarray 格式。
    label = np.array(label)

    # data(ndarray): [N,T_MAX,20,20,1], label(ndarray): [N,N_MOTION]
    return data, label


# 构建一个基于时间序列的卷积神经网络（CNN）和循环神经网络（GRU）的模型
def assemble_model(input_shape, n_class):
    model_input = Input(shape=input_shape, dtype='float32',
                        name='name_model_input')  # (@,T_MAX,20,20,1) # 定义模型的输入层，指定输入数据的形状和数据类型

    # Feature extraction part
    x = TimeDistributed(Conv2D(16, kernel_size=(5, 5), activation='relu', data_format='channels_last', \
                               input_shape=input_shape))(
        model_input)  # (@,T_MAX,20,20,1)=>(@,T_MAX,16,16,16) # 使用 TimeDistributed 将 Conv2D 层应用于每个时间步，提取特征。
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)  # (@,T_MAX,16,16,16)=>(@,T_MAX,8,8,16) 对特征图进行最大池化操作。
    x = TimeDistributed(Flatten())(x)  # (@,T_MAX,8,8,16)=>(@,T_MAX,8*8*16) 将特征图展平为一维向量。
    x = TimeDistributed(Dense(64, activation='relu'))(x)  # (@,T_MAX,8*8*16)=>(@,T_MAX,64) 全连接层，提取更高级的特征。
    x = TimeDistributed(Dropout(f_dropout_ratio))(x)  # 使用 Dropout 层进行正则化，防止过拟合。
    x = TimeDistributed(Dense(64, activation='relu'))(x)  # (@,T_MAX,64)=>(@,T_MAX,64)
    x = GRU(n_gru_hidden_units, return_sequences=False)(x)  # (@,T_MAX,64)=>(@,128)  # 使用 GRU 层进行时间序列建模，返回最后一个时间步的输出。
    x = Dropout(f_dropout_ratio)(x)
    model_output = Dense(n_class, activation='softmax', name='name_model_output')(
        x)  # (@,128)=>(@,n_class) # 定义输出层，使用 softmax 激活函数输出分类结果。

    # Model compiling
    model = Model(inputs=model_input, outputs=model_output)  # 定义模型的输入和输出。
    model.compile(optimizer=keras.optimizers.RMSprop(lr=f_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )  # 编译模型，指定优化器、损失函数和评估指标。
    return model


# ==============================================================
# Let's BEGIN >>>>
# if len(sys.argv) < 2:
#     print('Please specify GPU ...')
#     exit(0)
# if (sys.argv[1] == '1' or sys.argv[1] == '0'):
#     os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
#     config = tf.ConfigProto() # 创建 TensorFlow 的配置对象。
#     config.gpu_options.allow_growth = True # 设置 GPU 内存按需分配，避免一次性占用全部 GPU 内存。
#     set_session(tf.Session(config=config)) # 根据配置创建 TensorFlow 会话。
#     tf.set_random_seed(1) # 设置 TensorFlow 的随机种子，保证结果的可重复性。

# else:
#     print('Wrong GPU number, 0 or 1 supported!')
#     exit(0)

# Load data 调用 load_data 函数加载数据，返回处理后的数据 data 和标签 label。
data, label = load_data(data_dir, ALL_MOTION)
print('\nLoaded dataset of ' + str(label.shape[0]) + ' samples, each sized ' + str(data[0, :, :].shape) + '\n')

# Split train and test  使用 train_test_split 函数将数据和标签划分为训练集和测试集。
[data_train, data_test, label_train, label_test] = train_test_split(data, label, test_size=fraction_for_test)
print('\nTrain on ' + str(label_train.shape[0]) + ' samples\n' + \
      'Test on ' + str(label_test.shape[0]) + ' samples\n')

# One-hot encoding for train data 对训练集的标签进行独热编码处理。
label_train = onehot_encoding(label_train, N_MOTION)

# Load or fabricate model
if use_existing_model:
    model = load_model('model_widar3_trained.h5')
    model.summary()  # 打印模型的结构摘要，包括每一层的名称、输出形状和参数数量等信息。
else:
    model = assemble_model(input_shape=(T_MAX, 20, 20, 1), n_class=N_MOTION)
    model.summary()
    model.fit({'name_model_input': data_train}, {'name_model_output': label_train},
              batch_size=n_batch_size,
              epochs=n_epochs,
              verbose=1,
              validation_split=0.1, shuffle=True)  # 使用训练集数据进行模型训练，指定批量大小、训练轮数、验证集比例等参数。
    print('Saving trained model...')
    model.save('model_widar3_trained.h5')  # 训练完成后会保存模型为 'model_widar3_trained.h5' 文件。

# Testing...
print('Testing...')
label_test_pred = model.predict(data_test)  # 利用训练好的模型对测试集数据进行预测，得到预测结果。
label_test_pred = np.argmax(label_test_pred, axis=-1) + 1  # 将预测结果转换为类别索引（从0开始），并加1得到类别标签。

# Confusion Matrix 混淆矩阵
cm = confusion_matrix(label_test, label_test_pred)  # 使用混淆矩阵来比较真实标签和预测标签之间的结果。
print(cm)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 将混淆矩阵中的每个元素除以真实标签的总数，得到归一化后的混淆矩阵。
cm = np.around(cm, decimals=2)  # 将归一化后的混淆矩阵保留两位小数。
print(cm)

# Accuracy 准确率
test_accuracy = np.sum(label_test == label_test_pred) / (label_test.shape[0])  # 计算测试集的准确率，即预测正确的样本数除以总样本数。
print(test_accuracy)