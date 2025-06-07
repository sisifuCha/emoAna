# visualize.py
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from utils import TextDataset

import torch
from torch.utils.data import DataLoader




# 确保导入你的 TextDataset 类和 config
from main import TextDataset # 假设你的主文件叫 main.py
import config

# 设置中文字体，确保 simsun.ttc 字体文件路径正确
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
FONT_PATH = 'data/simsun.ttc' # 建议把字体文件放在data目录

def check_font():
    """检查字体文件是否存在"""
    if not os.path.exists(FONT_PATH):
        print(f"警告: 字体文件未找到于 '{FONT_PATH}'。词云图和中文标签可能无法正常显示。")
        return None
    return FONT_PATH

# (这里粘贴你原来的所有可视化函数)
# generate_wordclouds, mine_and_visualize_associations,
# visualize_word_distribution, visualize_sentiment_intensity

def generate_wordclouds(data, vocab, class_names):
    font_path = check_font()
    if not font_path: return

    id_to_word = {v: k for k, v in vocab.items()}
    class_word_freq = defaultdict(lambda: defaultdict(int))
    for words_line, label in data:
        for word_id in words_line:
            word = id_to_word.get(word_id, "UNK")
            if word in ("UNK", "PAD"): continue
            class_word_freq[label][word] += 1

    plt.figure(figsize=(15, 8))
    for i, class_name in enumerate(class_names):
        word_freq = class_word_freq[i]
        if not word_freq:
            print(f"类别 '{class_name}' 没有足够的数据生成词云。")
            continue
        wordcloud = WordCloud(
            width=800, height=400, background_color='white',
            font_path=font_path # 使用变量
        ).generate_from_frequencies(word_freq)
        plt.subplot(1, len(class_names), i + 1)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'"{class_name}" 类别词云')
        plt.axis('off')
    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, 'wordclouds.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"词云图已保存至: {save_path}")

# ... 其他可视化函数也粘贴到这里 ...
# 例如 mine_and_visualize_associations, visualize_word_distribution 等
# 确保它们内部的 font_path 和文件保存路径都正确使用了 config

# 关联规则挖掘与网络图可视化
def mine_and_visualize_associations(data, vocab, class_names, min_support=0.05):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    """
    挖掘关联规则并可视化
    :param data: 原始数据
    :param vocab: 词表
    :param class_names: 类别名称
    :param min_support: 最小支持度
    """
    # 反转vocab字典
    id_to_word = {v: k for k, v in vocab.items()}

    # 准备事务数据
    transactions = []
    for words_line, label in data:
        # 转换为词并添加类别标签
        words = [id_to_word.get(word_id, "UNK") for word_id in words_line]
        words.append(f"CLASS_{class_names[label]}")  # 添加类别标签
        transactions.append(words)

    # 转换为one-hot编码
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # 挖掘频繁项集
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    # 挖掘关联规则
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

    # 过滤出包含类别标签的规则
    class_related_rules = rules[
        rules['consequents'].apply(lambda x: any("CLASS_" in item for item in x)) |
        rules['antecedents'].apply(lambda x: any("CLASS_" in item for item in x))
        ]

    # 可视化关联规则网络图
    plt.figure(figsize=(12, 8))
    G = nx.DiGraph()

    # 添加节点和边
    for _, row in class_related_rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        weight = row['lift']

        # 添加节点
        for node in antecedents + consequents:
            G.add_node(node,
                       size=20 if "CLASS_" in node else 10,
                       color='red' if "CLASS_" in node else 'blue')

        # 添加边
        for ant in antecedents:
            for cons in consequents:
                G.add_edge(ant, cons, weight=weight)

    # 绘制网络图
    pos = nx.spring_layout(G, k=0.5)

    # 节点颜色和大小
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    node_sizes = [G.nodes[n]['size'] * 100 for n in G.nodes()]

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

    # 绘制边
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights)

    # 绘制标签
    labels = {n: n.replace("CLASS_", "") for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    plt.title("关联规则网络图 (边粗细表示lift值)")
    plt.axis('off')
    plt.savefig(os.path.join(config.RESULTS_DIR, 'association_network.png'), dpi=300)
    plt.close()

    return rules


# 情感词分布可视化
def visualize_word_distribution(data, vocab, class_names, sample_size=500):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    """
    可视化词汇在不同类别中的分布
    :param data: 原始数据
    :param vocab: 词表
    :param class_names: 类别名称
    :param sample_size: 采样大小
    """
    # 反转vocab字典
    id_to_word = {v: k for k, v in vocab.items()}

    # 收集词向量和标签
    words = []
    labels = []
    embeddings = []

    # 确保我们加载了预训练词向量
    if config.EMBEDDING_PRETRAINED is None:
        print("无法可视化词分布 - 未加载预训练词向量")
        return

    # 采样数据
    sampled_data = data[:sample_size]

    for words_line, label in sampled_data:
        for word_id in words_line[:10]:  # 只取每个样本的前10个词
            word = id_to_word.get(word_id, "UNK")
            if word == "UNK" or word == "PAD":
                continue

            words.append(word)
            labels.append(class_names[label])
            embeddings.append(config.EMBEDDING_PRETRAINED[word_id].cpu().numpy())

    embeddings = np.array(embeddings)

    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 创建DataFrame
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'word': words,
        'label': labels
    })

    # 绘制散点图
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x='x', y='y',
        hue='label',
        palette='viridis',
        alpha=0.7
    )

    # 添加一些重要词的标签
    for _, row in df.sample(20).iterrows():
        plt.text(row['x'], row['y'], row['word'],
                 fontsize=8, alpha=0.8)

    plt.title("词汇在情感类别中的分布 (t-SNE降维)")
    plt.legend(title='情感类别')
    plt.axis('off')
    plt.savefig(os.path.join(config.RESULTS_DIR, 'word_distribution.png'), dpi=300)
    plt.close()


# 情感强度可视化
def visualize_sentiment_intensity(model, data, vocab, class_names):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    """
    可视化情感强度分布
    :param model: 训练好的模型
    :param data: 测试数据
    :param vocab: 词表
    :param class_names: 类别名称
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in DataLoader(TextDataset(data), config.BATCH_SIZE):
            outputs = model(texts)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 创建DataFrame
    df = pd.DataFrame(all_probs, columns=class_names)
    df['True Label'] = [class_names[label] for label in all_labels]

    # 绘制情感强度分布
    plt.figure(figsize=(12, 6))
    for i, class_name in enumerate(class_names):
        plt.subplot(1, len(class_names), i + 1)
        sns.histplot(
            data=df[df['True Label'] == class_name],
            x=class_name,
            bins=20,
            kde=True
        )
        plt.title(f'"{class_name}"样本的情感强度分布')
        plt.xlabel('预测概率')

    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'sentiment_intensity.png'), dpi=300)
    plt.close()

