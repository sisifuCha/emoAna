# main.py

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, accuracy_score, f1_score, \
    recall_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# 从配置文件导入所有配置项
import config


# --- 核心功能函数 ---

def create_output_dirs():
    """检查并创建所有需要的输出目录"""
    print("--> 正在检查并创建输出目录...")
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print(f"    - 日志目录: {config.LOG_DIR}")
    print(f"    - 模型目录: {config.MODEL_DIR}")
    print(f"    - 结果目录: {config.OUTPUT_DIR}")


def is_chinese(uchar):
    """判断一个unicode字符是否为汉字"""
    return '\u4e00' <= uchar <= '\u9fa5'


def reserve_chinese(content):
    """只保留字符串中的汉字"""
    return "".join(filter(is_chinese, content))


def load_and_preprocess_data():
    """
    加载、预处理数据并进行划分 (修正版)
    """
    # 1. & 2. 加载和分词 (这部分完美，无需改动)
    data = pd.read_csv(
        config.DATA_PATH, sep='	', header=None,
        names=['label', 'separator', 'text'], usecols=['label', 'text'], encoding='utf-8'
    )
    data = data.dropna()
    stopwords = [line.strip() for line in open(config.STOPWORDS_PATH, 'r', encoding='utf-8').readlines()]
    data['tokenized_text'] = data['text'].apply(lambda x: [word for word in jieba.cut(x) if word not in stopwords])
    print(f"    - 成功加载 {len(data)} 条有效数据。")

    # 3. 标签编码 -> 得到整数标签 y_encoded
    le = LabelEncoder()
    y_encoded = le.fit_transform(data['label'])
    print(f"    - 发现 {len(le.classes_)} 个唯一标签: {le.classes_}")

    # 4. 文本序列化 -> 得到填充后的文本 X_padded
    tokenizer = Tokenizer(num_words=config.MAX_WORDS)
    tokenizer.fit_on_texts(data['tokenized_text'])
    X_sequences = tokenizer.texts_to_sequences(data['tokenized_text'])
    X_padded = pad_sequences(X_sequences, maxlen=config.MAX_LEN)

    # --- 【核心修正流程】 ---

    # 5. 先用【整数标签 y_encoded】进行分层划分，这样 stratify 才能工作
    #    注意，我们得到的 y 是临时的整数标签，所以命名为 y_*_int
    X_train_val, X_test, y_train_val_int, y_test_int = train_test_split(
        X_padded, y_encoded, test_size=0.1, random_state=config.RANDOM_SEED, stratify=y_encoded
    )
    X_train, X_val, y_train_int, y_val_int = train_test_split(
        X_train_val, y_train_val_int, test_size=1 / 9, random_state=config.RANDOM_SEED, stratify=y_train_val_int
    )

    # 6. 然后，对划分好的【整数标签】进行 One-Hot 编码，得到最终要用的标签
    num_classes = len(le.classes_)
    y_train = to_categorical(y_train_int, num_classes=num_classes)
    y_val = to_categorical(y_val_int, num_classes=num_classes)
    y_test = to_categorical(y_test_int, num_classes=num_classes)

    print(f"    - 数据集划分 -> 训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    print(f"    - 标签已转换为 One-Hot 格式，例如训练集第一个标签形状: {y_train[0].shape}")

    # 7. 返回【One-Hot编码后】的标签
    return X_train, X_val, X_test, y_train, y_val, y_test, tokenizer, le


def build_model(vocab_size, num_classes):
    """
    构建并编译RNN模型
    :param vocab_size: 词汇表大小 (从Tokenizer获得)
    :param num_classes: 类别数量 (从LabelEncoder获得)
    """
    print("--> Step2: 正在构建RNN模型...")

    inputs = Input(name='inputs', shape=[config.MAX_LEN])

    # 使用传入的 vocab_size 参数，而不是 config.MAX_WORDS
    layer = Embedding(vocab_size, config.EMBEDDING_DIM, input_length=config.MAX_LEN)(inputs)

    layer = SimpleRNN(config.RNN_UNITS)(layer)
    layer = Dense(config.DENSE_UNITS, activation="relu", name="FC1")(layer)
    layer = Dropout(config.DROPOUT_RATE)(layer)

    # 使用传入的 num_classes 参数，而不是 config.NUM_CLASSES
    outputs = Dense(num_classes, activation="softmax", name="FC2")(layer)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(
        loss=config.LOSS_FUNCTION,
        optimizer=config.OPTIMIZER,
        metrics=config.METRICS
    )
    return model

def train_model(model, train_data, train_labels, val_data, val_labels):
    """训练模型"""
    print("--> Step3: 正在训练模型...")

    callbacks = [
        TensorBoard(log_dir=config.LOG_DIR),
        EarlyStopping(monitor='val_loss', patience=3, verbose=1)  # 如果验证损失3个epoch不下降就停止
    ]

    model.fit(
        train_data, train_labels,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(val_data, val_labels),
        callbacks=callbacks
    )

    print("    - 模型训练完成。")
    model.save(config.MODEL_SAVE_PATH)
    print(f"    - 模型已保存至: {config.MODEL_SAVE_PATH}")
    return model


def evaluate_model(model, test_data, test_labels):
    """在测试集上评估模型并保存结果"""
    print("--> Step4: 正在评估模型...")

    # 加载模型（可选，用于验证保存的模型是否可用）
    # model = load_model(config.MODEL_SAVE_PATH)

    # 1. 预测
    test_pred_probs = model.predict(test_data)

    # 2. 将概率转换为类别标签
    pred_labels = np.argmax(test_pred_probs, axis=1)
    real_labels = np.argmax(test_labels, axis=1)

    # 3. 计算评估指标
    acc = accuracy_score(real_labels, pred_labels)
    precision = precision_score(real_labels, pred_labels, average='macro')
    recall = recall_score(real_labels, pred_labels, average='macro')
    f1 = f1_score(real_labels, pred_labels, average='macro')

    print("- -- 模型评估结果 - --")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print("--------------------")

    # 4. 绘制并保存混淆矩阵
    cm = confusion_matrix(real_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.CLASS_LABELS)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", values_format='d')
    plt.title("Confusion Matrix")
    plt.savefig(config.CONFUSION_MATRIX_SAVE_PATH, dpi=300)
    print(f"    - 混淆矩阵图已保存至: {config.CONFUSION_MATRIX_SAVE_PATH}")
    # plt.show() # 如果希望在运行时显示图像，取消此行注释

if __name__ == "__main__":
    # 创建目录
    create_output_dirs()

    # 1. 加载和预处理数据
    train_x, val_x, test_x, train_y, val_y, test_y, tokenizer, label_encoder = load_and_preprocess_data()

    # 2. 构建模型
    vocab_size = len(tokenizer.word_index) + 1  # 词汇表大小（总词数 + 1个用于padding的0）
    num_classes = len(label_encoder.classes_)  # 类别数量
    print(f"--> 参数确定: 词汇表大小={vocab_size}, 类别数量={num_classes}")
    rnn_model = build_model(vocab_size, num_classes)

    # 3. 训练模型
    trained_model = train_model(rnn_model, train_x, train_y, val_x, val_y)

    # 4. 评估模型
    evaluate_model(trained_model, test_x, test_y)

    print("- -- 所有流程执行完毕 - --")

