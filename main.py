# 主应用入口，用于加载 Streamlit 页面并调用各功能模块
import streamlit as st
from clustering.cluster import load_vector_data, cluster_vectors, visualize_clusters
from clustering.utils import parse_vector_str
from embeddings.extrator import extract_text_vectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import zipfile
import pandas as pd
from embeddings.token_cost import count_tokens, estimate_cost


# 设置页面宽度模式
st.set_page_config(layout="wide")

# 侧边栏功能选择器
st.sidebar.title("功能选择")
option = st.sidebar.radio("请选择要运行的功能模块：", ["提取文本向量", "HDBSCAN聚类" ,"向量可视化"])

# 聚类模块
if option == "HDBSCAN聚类":
    st.header("🔍 HDBSCAN 聚类分析")
    uploaded_file = st.file_uploader("请上传包含向量的 csv 文件", type=["csv", "parquet"])
    st.write("<h4 style='font-size: 14px;'>使用注释：</h4>", unsafe_allow_html=True)

    st.write('''```bash
            - 高相似度（>0.8）：设置较小的 min_cluster_size（例如 5），这样聚类会将相似的点合并到少数类别中。
- 中等相似度（0.5 - 0.8）：设置适中的 min_cluster_size（例如 10），聚类结果会有适中的类别数。
- 低相似度（<0.5）：设置较大的 min_cluster_size（例如 20），这样聚类会产生更多的类别，因为数据之间的相似度较低。''')
    min_cluster_size = st.text_input("最小聚类值(建议3~20)", value="5")  # 用户输入最小聚类尺寸
    submit2 = st.button('加载聚类')
    if uploaded_file is not None and submit2:
        with st.spinner('AI 正在加载并处理聚类，请稍候...'):
            df = load_vector_data(uploaded_file)
            df_clustered, model = cluster_vectors(df, int(min_cluster_size))
            vectors = np.array(df['vector'].tolist())
            similarity = cosine_similarity(vectors[:100])
            st.write(f'🔥 前100条向量平均相似度：{similarity.mean():.4f}')
            st.success("✅ 聚类完成，-1类 代表 噪声")
            st.dataframe(df_clustered)
            st.download_button("下载聚类结果", df_clustered.to_csv(index=False), file_name="clustered_result.csv")

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

            df, name_base = extract_text_vectors(uploaded_txt, openai_key, openai_model, chunk_size, chunk_overlap)

            # 获取所有文本段落
            text_list = df['text'].tolist()
            total_tokens = count_tokens(text_list, openai_model)
            estimated_usd = estimate_cost(total_tokens, openai_model)

            st.info(f"📊 估算 token 数量：{total_tokens}")
            st.info(f"💵 预计消耗金额：${estimated_usd:.4f} USD")

            # 下载 CSV 文件
            st.download_button("下载文本向量", df.to_csv(index=False), file_name=f'{name_base}.csv')





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

                if 'df' not in st.session_state:
                    st.session_state['df'] = []

                figs, pca_x, pca_y,tsne_x, tsne_y, umap_x, umap_y = visualize_clusters(df)
                x_lis = [pca_x, tsne_x, umap_x]
                y_lis = [pca_y, tsne_y, umap_y]
                tab_names = ["PCA", "t-SNE", "UMAP"]
                # 初始化 zip 文件缓冲区
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for i, (fig, name) in enumerate(zip(figs, tab_names), start=1):
                        # 将图片保存到临时缓冲区
                        img_buf = BytesIO()
                        fig.set_size_inches(4, 2.5)  # 控制图像大小
                        fig.savefig(img_buf, format="png", dpi=300)
                        img_buf.seek(0)

                        # 添加到 zip 包
                        zip_file.writestr(f"{name}_visualization.png", img_buf.read())

                # 展示 tab 页面
                tabs = st.tabs(tab_names)
                for tab, fig, name, x, y in zip(tabs, figs, tab_names, x_lis, y_lis):
                    with tab:
                        df_data = pd.DataFrame({
                            "x": x,
                            "y": y
                        })
                        st.scatter_chart(df_data, x="x", y="y")
                        st.divider()
                        st.pyplot(fig, use_container_width=True)
                        st.info(f"当前显示：{name} 降维图")

                # 提供 ZIP 下载按钮
                zip_buffer.seek(0)
                st.download_button(
                    label="📦 下载可视化图像集",
                    data=zip_buffer,
                    file_name="cluster_visualizations.zip",
                    mime="application/zip"
                )

