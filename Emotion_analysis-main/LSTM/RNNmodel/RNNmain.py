# main.py

import os
import numpy as np
import matplotlib.pyplot as plt
from jieba import lcut
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, accuracy_score, f1_score, \
    recall_score

from tensorflow.keras.models import Model, load_model
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
    """加载、清洗和预处理数据"""
    print("--> Step1: 正在加载和预处理数据...")

    # 1. 加载停用词
    with open(config.STOPWORDS_PATH, 'r', encoding='utf8') as f:
        stop_words = {line.strip() for line in f}

    # 2. 加载和解析数据
    all_words, all_labels = [], []
    try:
        # 推荐使用utf-8编码，如果确认是gbk，请改为'gbk'
        with open(config.DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    label, content = line.strip().split('	####	')
                    content_clean = reserve_chinese(content)
                    words = [word for word in lcut(content_clean) if word not in stop_words]
                    if words:
                        all_words.append(words)
                        all_labels.append(int(label))
                except ValueError:
                    print(f"    - 警告: 跳过格式不正确的行 -> {line.strip()}")
                    continue
    except FileNotFoundError:
        print(f"    - 错误: 数据文件未找到! 请检查路径: {config.DATA_PATH}")
        exit()  # 如果数据文件不存在，则直接退出程序

    print(f"    - 成功加载 {len(all_words)} 条有效数据。")

    # 3. 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_words, all_labels,
        test_size=config.VALIDATION_SPLIT_FROM_TOTAL,
        random_state=config.RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=config.TEST_SPLIT_FROM_VALIDATION,
        random_state=config.RANDOM_STATE
    )
    print(f"    - 数据集划分 -> 训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # 4. 标签编码 (Label Encoding + One-Hot Encoding)
    le = LabelEncoder()
    y_train_le = le.fit_transform(y_train).reshape(-1, 1)
    y_val_le = le.transform(y_val).reshape(-1, 1)
    y_test_le = le.transform(y_test).reshape(-1, 1)

    ohe = OneHotEncoder(sparse_output=False)  # sparse_output=False for newer scikit-learn
    y_train_ohe = ohe.fit_transform(y_train_le)
    y_val_ohe = ohe.transform(y_val_le)
    y_test_ohe = ohe.transform(y_test_le)

    # 5. 文本序列化 (Tokenization + Padding)
    tok = Tokenizer(num_words=config.MAX_WORDS)
    tok.fit_on_texts(X_train)  # 只在训练集上构建词汇表

    train_seq = tok.texts_to_sequences(X_train)
    val_seq = tok.texts_to_sequences(X_val)
    test_seq = tok.texts_to_sequences(X_test)

    train_seq_mat = pad_sequences(train_seq, maxlen=config.MAX_LEN)
    val_seq_mat = pad_sequences(val_seq, maxlen=config.MAX_LEN)
    test_seq_mat = pad_sequences(test_seq, maxlen=config.MAX_LEN)

    return train_seq_mat, val_seq_mat, test_seq_mat, y_train_ohe, y_val_ohe, y_test_ohe


def build_model():
    """构建并编译RNN模型"""
    print("--> Step2: 正在构建RNN模型...")

    inputs = Input(name='inputs', shape=[config.MAX_LEN])
    layer = Embedding(config.MAX_WORDS + 1, config.EMBEDDING_DIM, input_length=config.MAX_LEN)(inputs)
    layer = SimpleRNN(config.RNN_UNITS)(layer)
    layer = Dense(config.DENSE_UNITS, activation="relu", name="FC1")(layer)
    layer = Dropout(config.DROPOUT_RATE)(layer)
    outputs = Dense(config.NUM_CLASSES, activation="softmax", name="FC2")(layer)

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

    # --- 主程序入口 ---
    if __name__ == "__main__":
    # 0. 准备工作：创建目录
        create_output_dirs()

    # 1. 加载和预处理数据
    train_x, val_x, test_x, train_y, val_y, test_y = load_and_preprocess_data()

    # 2. 构建模型
    rnn_model = build_model()

    # 3. 训练模型
    trained_model = train_model(rnn_model, train_x, train_y, val_x, val_y)

    # 4. 评估模型
    evaluate_model(trained_model, test_x, test_y)

    print("- -- 所有流程执行完毕 - --")

