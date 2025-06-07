# -*- coding: utf-8 -*-
import os.path
import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
import config
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils import TextDataset

def get_data():
    """
        主函数，用于加载词表和数据集（训练集、验证集、测试集）。
    """
    # 定义分词器 (Tokenizer)：这里使用的是一个简单的字级别分词器。
    # lambda x: [y for y in x] 会将输入字符串 x 拆分成单个字符的列表。
    # 例如，"你好" -> ['你', '好']。这对于中文处理是一种常见策略。
    tokenizer = lambda x: [y for y in x]  # 字级别
    # 加载预先构建的词表。
    # 'rb' 表示以二进制只读模式打开文件。
    # vocab 应该是一个字典，键是字/词，值是对应的整数ID。
    vocab = pkl.load(open(config.VOCAB_PATH, 'rb'))
    # print('tokenizer',tokenizer)
    print('vocab',vocab)
    print(f"Vocab size: {len(vocab)}")

    train,dev,test = load_dataset(config.DATA_PATH, config.PAD_SIZE, tokenizer, vocab)
    return vocab, train, dev, test

def load_dataset(path, pad_size_from_config, tokenizer, vocab):
    '''
    将路径文本文件分词并转为三元组返回
    :param path: 文件路径
    :param pad_size: 每个序列的大小
    :param tokenizer: 转为字级别
    :param vocab: 词向量模型
    :return: 二元组，含有字ID，标签
    '''
    contents = []
    n=0
    with open(path, 'r', encoding='GBK') as f:
        # tqdm可以看进度条
        for line in tqdm(f):
            # 默认删除字符串line中的空格、’\n’、't’等。
            lin = line.strip()
            if not lin:
                continue
            # print(lin)
            label,content = lin.split('	####	')
            # word_line存储每个字的id
            words_line = []
            # 分割器，分词每个字
            token = tokenizer(content)
            # print(token)
            # 字的长度
            seq_len = len(token)
            if pad_size_from_config:
                # 如果字长度小于指定长度，则填充，否则截断
                if len(token) < pad_size_from_config:
                    token.extend([vocab.get(config.PAD)] * (pad_size_from_config - len(token)))
                else:
                    token = token[:pad_size_from_config]
                    seq_len = pad_size_from_config
            # 将每个字映射为ID
            # 如果在词表vocab中有word这个单词，那么就取出它的id；
            # 如果没有，就去除UNK（未知词）对应的id，其中UNK表示所有的未知词（out of vocab）都对应该id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(config.UNK)))
            n+=1
            contents.append((words_line, int(label)))

    # 数据集划分：
    # 使用 scikit-learn 的 train_test_split 函数将全部数据 (contents) 划分为训练集和临时集 (X_t)。
    # test_size=0.4 表示将40%的数据分给 X_t，剩下的60%作为训练集 (train)。
    # random_state=42 设置随机种子，确保每次划分结果一致，便于复现实验
    train, X_t = train_test_split(contents, test_size=0.4, random_state=config.RANDOM_SEED)
    # 再次使用 train_test_split 将临时集 (X_t) 划分为验证集 (dev) 和测试集 (test)。
    # test_size=0.5 表示将 X_t 的50%数据分给 test，剩下的50%作为验证集 (dev)。
    # 结合上一步，原始数据的 60% 为 train，20% 为 dev (0.4 * 0.5)，20% 为 test (0.4 * 0.5)。
    dev,test= train_test_split(X_t, test_size=0.5, random_state=config.RANDOM_SEED)
    return train,dev,test
# get_data()


# 以上是数据预处理的部分

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# 定义LSTM模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 使用预训练的词向量模型，freeze=False 表示允许参数在训练中更新
        # 在NLP任务中，当我们搭建网络时，第一层往往是嵌入层，对于嵌入层有两种方式初始化embedding向量，
        # 一种是直接随机初始化，另一种是使用预训练好的词向量初始化。
        self.embedding = nn.Embedding.from_pretrained(config.EMBEDDING_PRETRAINED, freeze=False)
        # bidirectional=True表示使用的是双向LSTM

        # - input_size=embed: 每个输入嵌入向量的大小 (词向量的维度)。
        # - hidden_size: LSTM 隐藏状态 (h) 中的特征数量。
        # - num_layers: 堆叠的循环 LSTM 层的数量。
        # - bidirectional=True: 使 LSTM 成为双向的。双向 LSTM 会从前向和后向两个方向处理序列，
        #   两个方向的输出通常会被拼接起来。这使得模型能同时从过去和未来的上下文中获取信息。
        # - batch_first=True: 指定输入和输出张量的第一个维度是批次大小
        #   (例如, [batch_size, sequence_length, features])。
        #   如果为 False (默认值), 则会是 [sequence_length, batch_size, features]。
        # - dropout: 如果非零，则在除最后一层外的每个 LSTM 层的输出上引入一个 Dropout 层，
        #   丢弃概率等于 'dropout'。这有助于防止过拟合。
        self.lstm = nn.LSTM(config.EMBED_DIM, config.HIDDEN_SIZE, config.NUM_LAYERS,
                            bidirectional=True, batch_first=True, dropout=config.DROPOUT)
        # 因为是双向LSTM，所以层数为config.hidden_size * 2
        self.fc = nn.Linear(config.HIDDEN_SIZE * 2, config.NUM_CLASSES)

    def forward(self, x):
        out = self.embedding(x)
        # lstm 的input为[batchsize, max_length, embedding_size]，输出表示为 output,(h_n,c_n),
        # 保存了每个时间步的输出，如果想要获取最后一个时间步的输出，则可以这么获取：output_last = output[:,-1,:]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# 权重初始化，默认xavier
# xavier和kaiming是两种初始化参数的方法
def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def plot_acc(train_acc):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_acc)))
    plt.plot(x, train_acc, alpha=0.9, linewidth=2, label='train acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend(loc='best')
    plt.title('Training Accuracy per Epoch')
    plt.savefig(os.path.join(config.RESULTS_DIR,'acc.png'), dpi=400)

def plot_loss(train_loss):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_loss)))
    plt.plot(x, train_loss, alpha=0.9, linewidth=2, label='train loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.title('Training Loss per Epoch')
    plt.savefig(os.path.join(config.RESULTS_DIR,'loss.png'), dpi=400)


# 定义训练的过程
def train( model, dataloaders):
    '''
    训练模型
    :param model: 模型
    :param dataloaders: 处理后的数据，包含trian,dev,test
    '''
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 损失函数
    loss_function = torch.nn.CrossEntropyLoss()

    dev_best_loss = float('inf')
    # gpu跑
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("========================================================================")
    print(f"开始训练 {device}...")
    print(f"总Epochs: {config.NUM_EPOCHS}, Batch Size: {config.BATCH_SIZE}, Learning Rate: {config.LEARNING_RATE}")
    print("========================================================================")
    plot_train_acc = []# 记录准确率和损失率
    plot_train_loss = []
    total_train_batches = len(dataloaders['train'])  # 每个epoch的总批次数，这非常重要！

    for i in range(config.NUM_EPOCHS):
        # 1，训练循环----------------------------------------------------------------
        # 将数据全部取完，记录每一个batch
        epoch_start_time = time.time()
        # 训练模式，可以更新参数
        model.train()
        print(f"---- Epoch{i + 1} / {config.NUM_EPOCHS} ----")
        train_lossi=0.
        train_acci = 0.
        for batch_idx ,(inputs, labels) in enumerate(dataloaders['train']):
            current_batch_in_epoch = batch_idx + 1
            # print(inputs.shape)
            inputs = torch.LongTensor(inputs).to(device)
            labels = torch.LongTensor(labels).to(device)
            # 梯度清零，防止累加
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # 获取当前批次的真实标签和预测标签 (用于计算准确率)
            true = labels.data.cpu()
            predic = torch.max(outputs.data, 1)[1].cpu()

            current_batch_loss = loss.item() # 当前批次的平均损失
            current_batch_acc = metrics.accuracy_score(true, predic) # 当前批次的准确率

            train_lossi += current_batch_loss
            train_acci += current_batch_acc
            if current_batch_in_epoch % config.log_batch_interval == 0 or current_batch_in_epoch == total_train_batches:
                print(f"  Epoch {i + 1} [批次 {current_batch_in_epoch:>4}/{total_train_batches}] "
                      f"当前批次损失: {current_batch_loss:.4f}, 当前批次准确率: {current_batch_acc:.2%}")
            # 2，验证集验证----------------------------------------------------------------

        # 计算当前epoch的平均训练损失和平均训练准确率
        avg_epoch_train_loss = train_lossi / total_train_batches
        avg_epoch_train_acc = train_acci / total_train_batches

        plot_train_loss.append(avg_epoch_train_loss)
        plot_train_acc.append(avg_epoch_train_acc)
        print(f"  正在验证 Epoch {i + 1}...")
        dev_acc, dev_loss = dev_eval(model, dataloaders['dev'], loss_function,Result_test=False)
        epoch_time_taken = get_time_dif(epoch_start_time) # 计算当前epoch耗时
        print(f"Epoch {i + 1}/{config.NUM_EPOCHS} | 耗时: {epoch_time_taken} | "
              f"训练损失: {avg_epoch_train_loss:.4f} | 训练准确率: {avg_epoch_train_acc:.2%} | "
              f"验证损失: {dev_loss:.4f} | 验证准确率: {dev_acc:.2%}")
        if dev_loss < dev_best_loss:
            print(f"  验证损失降低 ({dev_best_loss:.4f} --> {dev_loss:.4f})。正在保存模型到 {config.SAVE_PATH}")
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), config.SAVE_PATH)
    print("========================================================================")
    print("--- 训练完成 ---")
    print(f"达成的最佳验证损失: {dev_best_loss:.4f}")
    print(f"模型已保存到: {config.SAVE_PATH}")
    print("========================================================================")
    plot_acc(plot_train_acc)
    plot_loss(plot_train_loss)
    # 3，验证循环----------------------------------------------------------------
    print("开始在测试集上进行最终评估......")
    model.load_state_dict(torch.load(config.SAVE_PATH))
    model.eval()
    start_time = time.time()
    test_acc, test_loss = dev_eval(model, dataloaders['test'], loss_function,Result_test=True)
    test_time_taken = get_time_dif(start_time)
    print("======================= 测试集评估结果 ===================================")
    print(f"测试耗时: {test_time_taken}")
    # result_test 函数（在dev_eval中被调用）会打印详细的 acc, precision, recall, f1
    # 这里只打印一个概要的损失和准确率 (如果 dev_eval 返回了 acc)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.2%}")  # 假设 dev_eval 返回了准确率
    print("========================================================================")

def result_test(real, pred):
    cv_conf = confusion_matrix(real, pred)
    acc = accuracy_score(real, pred)
    precision = precision_score(real, pred, average='micro')
    recall = recall_score(real, pred, average='micro')
    f1 = f1_score(real, pred, average='micro')
    patten = 'test:  acc: %.4f   precision: %.4f   recall: %.4f   f1: %.4f'
    print(patten % (acc, precision, recall, f1,))
    labels11 = ['negative', 'active']
    disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels11)
    disp.plot(cmap="Blues", values_format='')
    # --- 新增代码：关闭网格线 ---
    if hasattr(disp, 'ax_') and disp.ax_ is not None:
        disp.ax_.grid(False)
    # --------------------------
    plt.savefig(os.path.join(config.RESULTS_DIR,'reConfusionMatrix.tif'), dpi=400)
    plt.close()


def dev_eval(model, data, loss_function, Result_test=False):
    '''
    :param model: 模型
    :param data: 验证集集或者测试集的数据
    :param loss_function: 损失函数
    :return: 损失和准确率
    '''
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    # 这一行很重要：自动获取模型所在的设备 (cpu or cuda)
    device = next(model.parameters()).device

    with torch.no_grad():
        # 这里的 texts 和 labels 是从 DataLoader 出来的 Python 列表
        for texts, labels in data:
            # --- 【核心修改点】---
            # 1. 将 Python 列表转换为 PyTorch Tensor
            # 2. 将 Tensor 移动到模型所在的正确设备上
            texts_tensor = torch.LongTensor(texts).to(device)
            labels_tensor = torch.LongTensor(labels).to(device)

            outputs = model(texts_tensor)
            loss = loss_function(outputs, labels_tensor)
            loss_total += loss.item()

            # 将 tensor 转换回 numpy array 以便进行评估
            # 注意：这里也要使用新的 labels_tensor
            labels_np = labels_tensor.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()

            labels_all = np.append(labels_all, labels_np)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if Result_test:
        # 假设 result_test 是一个你已经定义好的函数
        result_test(labels_all, predict_all)

    return acc, loss_total / len(data)


if __name__ == '__main__':
    # 确保输出目录存在
    # (如果config.py中有定义，请确保路径正确)
    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)
    if not os.path.exists(os.path.dirname(config.SAVE_PATH)):
        os.makedirs(os.path.dirname(config.SAVE_PATH))

    # 设置随机数种子，保证每次运行结果一致
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    torch.cuda.manual_seed_all(config.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    # --- 从 config.py 读取运行模式 ---
    print("=" * 60)
    print(f"--> 当前运行模式 (来自 config.py): '{config.RUN_MODE}'")
    print("=" * 60)

    start_time = time.time()
    print("--> 正在加载数据...")
    vocab, train_data, dev_data, test_data = get_data()

    # 读取类别名称
    class_names_path = 'data/class.txt'
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f]
    else:
        print(f"警告: 类别文件 {class_names_path} 未找到，将使用默认标签。")
        class_names = ['类别0', '类别1']  # 提供一个默认值

    time_dif = get_time_dif(start_time)
    print(f"--> 数据加载完成，耗时: {time_dif}")

    # --- 根据 config.RUN_MODE 的值选择执行任务 ---

    # 如果模式是 'train' 或 'all'，则执行训练流程
    if config.RUN_MODE in ['train', 'all']:
        print("          开始进入模型训练与评估流程")
        print("=" * 50)

        dataloaders = {
            'train': DataLoader(TextDataset(train_data), config.BATCH_SIZE, shuffle=True),
            'dev': DataLoader(TextDataset(dev_data), config.BATCH_SIZE, shuffle=True),
            'test': DataLoader(TextDataset(test_data), config.BATCH_SIZE, shuffle=True)
        }
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = Model().to(device)
        init_network(model)
        train(model, dataloaders)

    # 如果模式是 'visualize' 或 'all'，则执行可视化流程
    if config.RUN_MODE in ['visualize', 'all']:
        print("          开始进入数据可视化流程")
        print("=" * 50)

        # 动态导入可视化模块，确保它只在需要时被加载
        try:
            import visualize
        except ImportError:
            print("错误: 无法导入 visualize.py。请确保该文件存在于同一目录下。")
            exit()  # 如果找不到可视化文件，直接退出

        print("--> 正在生成词云图...")
        visualize.generate_wordclouds(train_data, vocab, class_names)

        # 注意：以下可视化非常耗时，可以根据需要注释掉
        print("--> 正在进行关联规则挖掘与可视化(这可能需要很长时间)...")
        visualize.mine_and_visualize_associations(train_data, vocab, class_names)

        print("--> 正在进行词分布可视化(t - SNE)...")
        visualize.visualize_word_distribution(train_data, vocab, class_names)

        # 情感强度可视化需要已训练好的模型
        if os.path.exists(config.SAVE_PATH):
            print("--> 正在进行情感强度可视化...")
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            model = Model().to(device)
            model.load_state_dict(torch.load(config.SAVE_PATH))
            visualize.visualize_sentiment_intensity(model, test_data, vocab, class_names)
        else:
             print("--> 跳过情感强度可视化，因为找不到已训练的模型。")

    print("所有任务执行完毕。")