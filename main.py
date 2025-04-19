# 包含：向量数据加载、聚类执行、TSNE 可视化函数
import platform
import matplotlib.font_manager as fm
import urllib.request

import pandas as pd
import numpy as np
import hdbscan
import os
from clustering.utils import parse_vector_str
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap




# 加载用户上传的数据文件，支持csv
def load_vector_data(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(uploaded_file)
        df['vector'] = df['vector'].apply(parse_vector_str)
    else:
        raise ValueError("不支持的文件格式，请上传.csv 文件")
    return df

# 执行 HDBSCAN 聚类分析
def cluster_vectors(df, min_cluster_size=5):
    vectors = np.array(df['vector'].tolist())  # 转换为 numpy array
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)  # 实例化聚类器
    labels = clusterer.fit_predict(vectors)  # 聚类
    df['cluster'] = labels  # 添加聚类标签列
    return df, clusterer

# 使用t-SNE（t-分布邻域嵌入）降维可视化聚类结果



def visualize_clusters(df):
    system = platform.system()

    if system == "Linux":
        # 在 Linux 系统中安装 SimHei 字体
        font_url = "https://github.com/owent-utils/fonts/raw/master/ttf/SimHei.ttf"
        font_dir = "/usr/share/fonts/truetype/simhei/"
        font_path = os.path.join(font_dir, "SimHei.ttf")

        # 创建字体目录
        if not os.path.exists(font_dir):
            os.makedirs(font_dir)

        # 下载 SimHei 字体
        print("正在下载 SimHei 字体...")
        urllib.request.urlretrieve(font_url, font_path)
        print(f"SimHei 字体已安装至 {font_path}")

        # 更新字体缓存
        fm.fontManager._rebuild()

        # 设置 Matplotlib 字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

    elif system == "Darwin":
        # 在 macOS 中安装字体
        font_url = "https://github.com/owent-utils/fonts/raw/master/ttf/SimHei.ttf"
        font_dir = os.path.expanduser("~/Library/Fonts/")
        font_path = os.path.join(font_dir, "SimHei.ttf")

        if not os.path.exists(font_dir):
            os.makedirs(font_dir)

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    vectors = np.array(df['vector'].tolist())
    n_samples = len(vectors)
    n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'].unique() else 0)

    # 动态调整 TSNE perplexity
    perplexity = min(30, max(2, n_samples // 3))

    # 1. PCA 降维
    pca_result = PCA(n_components=2).fit_transform(vectors)
    df['pca_x'] = pca_result[:, 0]
    df['pca_y'] = pca_result[:, 1]

    # 2. t-SNE 降维
    tsne_result = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(vectors)
    df['tsne_x'] = tsne_result[:, 0]
    df['tsne_y'] = tsne_result[:, 1]

    # 3. UMAP 降维
    umap_result = umap.UMAP(n_components=2, random_state=42).fit_transform(vectors)
    df['umap_x'] = umap_result[:, 0]
    df['umap_y'] = umap_result[:, 1]

    # 创建图形对象
    figs = []

    subtitle = f"样本数：{n_samples}，聚类数：{n_clusters}"

    # PCA 图
    fig1 = plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='pca_x', y='pca_y', hue='cluster', palette='tab10')
    plt.title(f"HDBSCAN 聚类可视化（PCA降维）\n{subtitle}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    figs.append(fig1)

    # TSNE 图
    fig2 = plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='tsne_x', y='tsne_y', hue='cluster', palette='tab10')
    plt.title(f"HDBSCAN 聚类可视化（TSNE降维）\n{subtitle}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    figs.append(fig2)

    # UMAP 图
    fig3 = plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='umap_x', y='umap_y', hue='cluster', palette='tab10')
    plt.title(f"HDBSCAN 聚类可视化（UMAP降维）\n{subtitle}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    figs.append(fig3)

    return figs
