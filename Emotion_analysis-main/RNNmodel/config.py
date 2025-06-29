# config.py

import os

# --- 1. 文件与路径配置 ---
# os.path.abspath(__file__) 获取当前config.py的绝对路径
# os.path.dirname(...) 获取该文件所在的目录路径 (即 .../RNNmodel)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据路径
# 'data' 文件夹是 BASE_DIR 的直接子目录
DATA_DIR = os.path.join(BASE_DIR, 'data') # <--- 修改点 1: 移除了多余的 'RNNmodel/'

# 停用词文件和数据文件都在 DATA_DIR 中
STOPWORDS_PATH = os.path.join(DATA_DIR, 'stopwords.txt') # <--- 修改点 2: 基于正确的DATA_DIR构建
DATA_PATH = os.path.join(DATA_DIR, 'finaldata.txt')

# 输出目录 (这些是正确的，因为它们是 BASE_DIR 的直接子目录)
LOG_DIR = os.path.join(BASE_DIR, 'logs')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# 模型和图表文件名
MODEL_NAME = 'RNN_Sentiment_Model.h5'
CONFUSION_MATRIX_NAME = 'Confusion_Matrix.png'

# 最终保存路径
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
CONFUSION_MATRIX_SAVE_PATH = os.path.join(OUTPUT_DIR, CONFUSION_MATRIX_NAME)


# --- 2. 数据预处理配置 ---
# 数据集划分比例 (Train: 60%, Validation: 20%, Test: 20%)
VALIDATION_SPLIT_FROM_TOTAL = 0.2 # 先从总数据中分出40%作为（验证集+测试集）
TEST_SPLIT_FROM_VALIDATION = 0.5 # 再从这40%中分出一半（即总数据的20%）作为测试集
RANDOM_STATE = 42 # 随机种子，确保每次划分结果一致


# --- 3. Tokenizer 和序列配置 ---
MAX_WORDS = 5000  # 词汇表中保留的最大词数
MAX_LEN = 150     # 每个文本序列的最大长度


# --- 4. 模型超参数配置 ---
EMBEDDING_DIM = 128   # 词嵌入维度
RNN_UNITS = 128       # RNN层单元数
DENSE_UNITS = 128     # 全连接层单元数
DROPOUT_RATE = 0.5    # Dropout比率
NUM_CLASSES = 2       # 分类类别数 (正面/负面)


# --- 5. 训练配置 ---
OPTIMIZER = 'RMSprop'
LOSS_FUNCTION = 'categorical_crossentropy'
METRICS = ['accuracy']
BATCH_SIZE = 128
EPOCHS = 10
RANDOM_SEED=40


# --- 6. 评估配置 ---
# 混淆矩阵的标签 (注意顺序，0对应'negative', 1对应'positive')
CLASS_LABELS = ['negative', 'positive'] # 'active' 可能是一个笔误，改为 'positive' 更标准
