# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from jieba import lcut
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, accuracy_score, f1_score, \
    recall_score

# 使用 tensorflow.keras 而不是独立的 tensorflow.python.keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import Callback, TensorBoard


# --- 数据处理函数 (与您原来的一致) ---
def is_chinese(uchar):
    if (uchar >= '\u4e00' and uchar <= '\u9fa5'):
        return True
    else:
        return False


def reserve_chinese(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str += i
    return content_str


def dataParse(text, stop_words):
    # 假设数据文件编码为 utf-8，如果不是请修改
    label, content, = text.split('	####	')
    content = reserve_chinese(content)
    words = lcut(content)
    words = [i for i in words if not i in stop_words]
    return words, int(label)


def getStopWords():
    # 假设停用词文件在 ./data/stopwords.txt
    file = open('./data/stopwords.txt', 'r', encoding='utf8')
    words = [i.strip() for i in file.readlines()]
    file.close()
    return words


def getFormatData():
    # 假设数据文件在 ./data/finaldata.txt
    # 注意：您的原始代码使用了 'gbk'，但 'utf-8' 更常用。请根据您的文件实际编码修改。
    try:
        with open('./data/finaldata.txt', 'r', encoding='utf-8') as file:
            texts = file.readlines()
    except FileNotFoundError:
        print("错误: 数据文件未找到! 请确保 './data/finaldata.txt' 文件存在。")
        exit()  # 如果文件不存在，直接退出

    stop_words = getStopWords()
    all_words = []
    all_labels = []
    for text in texts:
        content, label = dataParse(text, stop_words)
        if len(content) <= 0:
            continue
        all_words.append(content)
        all_labels.append(label)
    return all_words, all_labels


# --- 开始执行 ---

## 读取和预处理数据集
print("--- Step 1: 正在加载和预处理数据 ---")
data, label = getFormatData()
print(f"原始数据加载完成，总样本数: {len(data)}")

X_train, X_t, train_y, v_y = train_test_split(data, label, test_size=0.4, random_state=42)
X_val, X_test, val_y, test_y = train_test_split(X_t, v_y, test_size=0.5, random_state=42)
print(f"数据集划分 -> 训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

## 对数据集的标签数据进行编码
le = LabelEncoder()
train_y = le.fit_transform(train_y).reshape(-1, 1)
val_y = le.transform(val_y).reshape(-1, 1)
test_y = le.transform(test_y).reshape(-1, 1)

## 对数据集的标签数据进行one-hot编码
ohe = OneHotEncoder(sparse_output=False)
train_y = ohe.fit_transform(train_y)
val_y = ohe.transform(val_y)
test_y = ohe.transform(test_y)
print("标签编码和One-Hot编码完成。")
print(f"训练集标签Shape: {train_y.shape}")

## 使用Tokenizer对词组进行编码
max_words = 5000
max_len = 600
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)  # 仅在训练集上拟合

train_seq = tok.texts_to_sequences(X_train)
val_seq = tok.texts_to_sequences(X_val)
test_seq = tok.texts_to_sequences(X_test)

## 将每个序列调整为相同的长度
train_seq_mat = pad_sequences(train_seq, maxlen=max_len)
val_seq_mat = pad_sequences(val_seq, maxlen=max_len)
test_seq_mat = pad_sequences(test_seq, maxlen=max_len)
print("文本序列化和填充完成。")
print(f"训练集数据Shape: {train_seq_mat.shape}")

## 定义RNN模型
print("- -- Step2: 正在构建RNN模型 - --")
inputs = Input(name='inputs', shape=[max_len])
layer = Embedding(max_words + 1, 128, input_length=max_len)(inputs)
layer = SimpleRNN(128)(layer)
layer = Dense(128, activation="relu", name="FC1")(layer)
layer = Dropout(0.5)(layer)
layer = Dense(2, activation="softmax", name="FC2")(layer)
model = Model(inputs=inputs, outputs=layer)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=["accuracy"])

# ==================== 新增部分：自定义回调 ====================


class EpochLoggerCallback(Callback):
    """一个在每个epoch结束后打印详细日志的回调。"""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # 从logs字典获取指标，如果不存在则显示 'N/A'
        loss = logs.get('loss', 'N/A')
        accuracy = logs.get('accuracy', 'N/A')
        val_loss = logs.get('val_loss', 'N/A')
        val_accuracy = logs.get('val_accuracy', 'N/A')

        print("\n" + "=" * 80)
        # epoch是从0开始的，所以我们打印 epoch + 1
        print(f"Epoch {epoch + 1} 完成 - 调试信息:")
        print(f"    - 训练损失 (Training Loss):       {loss:.4f}")
        print(f"    - 训练准确率 (Training Accuracy):   {accuracy:.4f}")
        print(f"    - 验证损失 (Validation Loss):     {val_loss:.4f}")
        print(f"    - 验证准确率 (Validation Accuracy): {val_accuracy:.4f}")
        print("=" * 80)


# 实例化我们的自定义回调
epoch_logger = EpochLoggerCallback()
# ==========================================================

## 模型训练
print("- -- Step3: 正在开始模型训练 - --")
model_fit = model.fit(
    train_seq_mat, train_y,
    batch_size=128,
    epochs=10,
    validation_data=(val_seq_mat, val_y),
    # 在回调列表中加入我们自定义的 logger
    callbacks=[
        TensorBoard(log_dir='./logs'),  # 建议将日志放入 'logs' 文件夹
        epoch_logger
    ]
)

print("- -- Step4: 训练完成，正在评估模型 - --")

# 检查并创建模型保存目录
model_dir = './model_output'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, 'RNN.h5')

# 保存模型
model.save(model_path)
print(f"模型已保存至: {model_path}")

# 注意：这里直接使用训练好的 `model` 对象进行预测即可，
# `del model` 和 `load_model` 在这里是多余的，除非你想验证模型的保存和加载功能。
# 为了简洁，我们直接使用内存中的模型。

## 对测试集进行预测
test_pre = model.predict(test_seq_mat)

pred = np.argmax(test_pre, axis=1)
real = np.argmax(test_y, axis=1)

# 计算评估指标
acc = accuracy_score(real, pred)
# 对于二分类或多分类，使用 'macro' 或 'weighted' 平均更具信息量
precision = precision_score(real, pred, average='macro')
recall = recall_score(real, pred, average='macro')
f1 = f1_score(real, pred, average='macro')

print("- -- 模型在测试集上的表现 - --")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {precision:.4f} (Macro Avg)")
print(f"  Recall:    {recall:.4f} (Macro Avg)")
print(f"  F1-Score:  {f1:.4f} (Macro Avg)")
print("--------------------------")

## 绘制并保存混淆矩阵
cv_conf = confusion_matrix(real, pred)
# 您的标签是0和1，对应 "negative" 和 "positive"（"active"可能是笔误）
labels_display = ['negative', 'positive']
disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels_display)
disp.plot(cmap="Blues", values_format='d')

# 将图表也保存到输出目录
plt.savefig(os.path.join(model_dir, "ConfusionMatrix.png"), dpi=300)
print(f"混淆矩阵图已保存至: {os.path.join(model_dir, 'ConfusionMatrix.png')}")
plt.show()  # 在屏幕上显示图像
