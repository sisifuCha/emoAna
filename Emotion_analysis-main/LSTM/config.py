# config.py
import torch
import numpy as np
import os

# --- 路径配置 (Path Configurations) ---
# 建议使用相对路径或基于项目根目录的路径，使其更具可移植性
# 如果你的 data 和 saved_dict 文件夹与 config.py 和主脚本在同一级或其子目录：
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # 获取 config.py 所在的目录

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SAVE_DIR = os.path.join(PROJECT_ROOT, 'saved_dict')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results') # 假设你的 results 文件夹也在这里

# 确保这些目录存在，如果不存在则创建 (可选，但推荐)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


DATA_PATH = os.path.join(DATA_DIR, 'data.txt')              # 数据集
VOCAB_PATH = os.path.join(DATA_DIR, 'vocab.pkl')            # 词表
EMBEDDING_PATH = os.path.join(DATA_DIR, 'embedding_Tencent.npz') # 预训练词向量路径
SAVE_PATH = os.path.join(SAVE_DIR, 'lstm.ckpt')        # 模型训练结果

# --- 预训练词向量配置 (Pretrained Embedding Configuration) ---
# 尝试加载预训练词向量，如果文件不存在，则可以设置为 None 或抛出错误
try:
    embedding_pretrained_np = np.load(EMBEDDING_PATH)["embeddings"].astype('float32')
    EMBEDDING_PRETRAINED = torch.tensor(embedding_pretrained_np)
    EMBED_DIM = EMBEDDING_PRETRAINED.size(1)        # 词向量维度 (从加载的词向量推断)
except FileNotFoundError:
    print(f"警告: 预训练词向量文件未找到于 {EMBEDDING_PATH}. 将 EMBEDDING_PRETRAINED 设置为 None，EMBED_DIM 需要手动设置或后续处理。")
    EMBEDDING_PRETRAINED = None
    EMBED_DIM = 300 # 如果预训练文件不存在，你需要提供一个默认的嵌入维度或在模型中处理这种情况
except KeyError:
    print(f"警告: 在 {EMBEDDING_PATH} 中未找到键 'embeddings'. 将 EMBEDDING_PRETRAINED 设置为 None。")
    EMBEDDING_PRETRAINED = None
    EMBED_DIM = 300


RUN_MODE = 'visualize'
# --- 模型超参数 (Model Hyperparameters) ---
DROPOUT = 0.5                               # 随机丢弃率
NUM_CLASSES = 2                             # 类别数 (例如：正面/负面)
HIDDEN_SIZE = 128                           # LSTM 隐藏层大小
NUM_LAYERS = 2                              # LSTM 层数

# --- 训练超参数 (Training Hyperparameters) ---
NUM_EPOCHS = 3                              # 训练轮次 (epoch数)
BATCH_SIZE = 128                            # mini-batch大小
LEARNING_RATE = 1e-3                        # 学习率

# --- 数据处理参数 (Data Processing Parameters) ---
PAD_SIZE = 50                               # 每句话处理成的长度 (短填长切)
MAX_VOCAB_SIZE = 10000                      # 词表长度限制 (如果需要构建词表时使用)
UNK, PAD = '<UNK>', '<PAD>'                 # 未知字，padding符号

# --- 其他设置 (Other Settings) ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 1                             # 随机种子，用于复现性
log_batch_interval=64

# --- 绘图和结果保存相关 ---
ACC_PLOT_PATH = os.path.join(RESULTS_DIR, 'acc.png')
LOSS_PLOT_PATH = os.path.join(RESULTS_DIR, 'loss.png')
CONFUSION_MATRIX_PLOT_PATH = os.path.join(RESULTS_DIR, 'reConfusionMatrix.tif')
PLOT_LABELS = ['negative', 'active'] # 用于混淆矩阵的标签

