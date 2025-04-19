# 主应用入口，用于加载 Streamlit 页面并调用各功能模块
import streamlit as st
from clustering.cluster import load_vector_data, cluster_vectors, visualize_clusters
from clustering.utils import parse_vector_str
from embeddings.extrator import extract_text_vectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import zipfile



# 设置页面宽度模式
st.set_page_config(layout="wide")

# 侧边栏功能选择器
st.sidebar.title("功能选择")
option = st.sidebar.radio("请选择要运行的功能模块：", ["提取文本向量", "HDBSCAN聚类" ,"向量可视化"])

# 聚类模块
if option == "HDBSCAN聚类":
    st.header("🔍 HDBSCAN 聚类分析")
    uploaded_file = st.file_uploader("请上传包含向量的 csv 文件", type=["csv", "parquet"])
    min_cluster_size = st.text_input("最小聚类大小(建议3~10)", value="5")  # 用户输入最小聚类尺寸
    submit2 = st.button('加载聚类')
    if uploaded_file is not None and submit2:
        with st.spinner('AI 正在加载并处理聚类，请稍候...'):
            df = load_vector_data(uploaded_file)
            df_clustered, model = cluster_vectors(df, int(min_cluster_size))
            vectors = np.array(df['vector'].tolist())
            similarity = cosine_similarity(vectors[:100])
            st.write(f"🔥 前100条向量平均相似度：{similarity.mean():.4f}\n高相似度（>0.8）：设置较小的 min_cluster_size（例如 5），这样聚类会将相似的点合并到少数类别中。\n中等相似度（0.5 ≤ avg_similarity < 0.8）：设置适中的 min_cluster_size（例如 10），聚类结果会有适中的类别数。\n低相似度（<0.5）：设置较大的 min_cluster_size（例如 20），这样聚类会产生更多的类别，因为数据之间的相似度较低。")
            st.success("✅ 聚类完成，-1类 代表 噪声")
            st.dataframe(df_clustered)
            st.download_button("下载聚类结果为 CSV", df_clustered.to_csv(index=False), file_name="clustered_result.csv")

# 文本向量生成模块
elif option == "提取文本向量":
    st.header("📄 文本向量生成")
    st.write('###### 注：需科学上网')
    openai_key = st.text_input("🔑请输入你的 OpenAI API Key：", type="password")
    st.markdown('[OpenAI Key获取方式](https://platform.openai.com/api-keys)')
    uploaded_txt = st.file_uploader("📄 上传 .txt 文本文件", type="txt")

    # 用户选择OpenAI嵌入模型
    openai_model = st.selectbox(
        "请选择OpenAI嵌入模型：",
        [
            "text-embedding-ada-002(推荐模型，速度快且成本低)",
            "text-embedding-babbage-001(适用于稍复杂的任务)",
            "text-embedding-curie-001(更高质量但更高成本的模型)"
        ]
    )

    # 用户选择 chunk_size 和 chunk_overlap
    chunk_size = st.number_input("输入每段的最大字符数 (chunk_size)", min_value=100, max_value=1000, value=300, step=50)
    chunk_overlap = st.number_input("输入段落的重叠字符数 (chunk_overlap)", min_value=10, max_value=500, value=50, step=10)

    submit1 = st.button('提取文本向量')
    if uploaded_txt and openai_key and submit1:
        with st.spinner("AI 正在提取文本向量..."):
            # 传递所选的模型名和用户输入的 chunk_size 与 chunk_overlap
            df, name_base = extract_text_vectors(uploaded_txt, openai_key, openai_model, chunk_size, chunk_overlap)

            # 下载 CSV 文件
            st.download_button("下载文本向量为 CSV", df.to_csv(index=False), file_name=f'{name_base}.csv')





elif option == "向量可视化":
    st.header("📊 向量聚类可视化")
    uploaded_file = st.file_uploader("请上传带有聚类标签和向量的 csv 文件", type="csv")
    submit3 = st.button('可视化加载')
    if uploaded_file and submit3:
        with st.spinner('AI 正在生成可视化图像...'):
            df = load_vector_data(uploaded_file)
            # 读取文件后检查数据
            df['vector'] = df['vector'].apply(parse_vector_str)
            df = df[df['vector'].apply(len) > 0]  # 过滤掉空的向量

            if 'cluster' not in df.columns:
                st.warning("该文件不包含 'cluster' 列，请先执行聚类任务。")
            else:
                # 获取图像对象

                figs = visualize_clusters(df)
                tab_names = ["PCA", "t-SNE", "UMAP"]

                # 初始化 zip 文件缓冲区
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for i, (fig, name) in enumerate(zip(figs, tab_names), start=1):
                        # 将图片保存到临时缓冲区
                        img_buf = BytesIO()
                        fig.set_size_inches(6, 4)  # 控制图像大小
                        fig.savefig(img_buf, format="png", dpi=300, bbox_inches='tight')
                        img_buf.seek(0)

                        # 添加到 zip 包
                        zip_file.writestr(f"{name}_visualization.png", img_buf.read())

                # 展示 tab 页面
                tabs = st.tabs(tab_names)
                for tab, fig, name in zip(tabs, figs, tab_names):
                    with tab:
                        st.pyplot(fig, use_container_width=True)
                        st.info(f"当前显示：{name} 降维图")

                # 提供 ZIP 下载按钮
                zip_buffer.seek(0)
                st.download_button(
                    label="📦 下载所有可视化图像为 ZIP",
                    data=zip_buffer,
                    file_name="cluster_visualizations.zip",
                    mime="application/zip"
                )

