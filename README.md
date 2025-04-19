 
# 🧠# Vector Clustering & Visualization



本项目提供一个交互式 Web 界面，用于从文本中提取嵌入向量、进行 HDBSCAN 聚类分析，并使用 t-SNE 进行可视化。

This project provides an interactive Streamlit app for text embedding extraction, HDBSCAN clustering, and TSNE-based visualization.




## 🚀 Features

### 1. 文本向量提取 | Text Embedding
- Utilize OpenAI embedding models: text-embedding-ada-002, babbage-001, curie-001
- Support .txt file upload and configurable text chunking
- Output vector data in both CSV and Parquet formats

### 2. 向量聚类 | Vector Clustering
- Support ```.csv```/ ```.parquet``` file uploads
- Perform automatic clustering using HDBSCAN
- Display average similarity and enable result download

### 3. 向量可视化 | Vector Visualization
- Reduce dimensions using ```PCA```,```t-SNE```,```u-map```
- Visualize clustering results and download the image
