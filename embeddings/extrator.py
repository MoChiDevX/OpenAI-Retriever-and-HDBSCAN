import pandas as pd
import uuid
import tempfile
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 文本向量提取函数
def extract_text_vectors(uploaded_txt, openai_key, model_option, chunk_size, chunk_overlap):
    # 检查上传文件大小，避免加载过大的文件
    file_size = uploaded_txt.size
    if file_size > 50 * 1024 * 1024:  # 超过 50MB
        raise ValueError("上传文件过大，请选择较小的文件进行上传。")

    # 保存上传的 txt 文件为临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as tmp:
        tmp.write(uploaded_txt.read().decode('utf-8'))
        tmp_path = tmp.name  # 获取临时路径

    # 使用 RecursiveCharacterTextSplitter 拆分文本
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '。', '！', '？', '，', '、', '.', '!', '?', ',', ';', ':'],  # 包含中文和英文标点
        chunk_size=chunk_size,  # 使用用户输入的 chunk_size
        chunk_overlap=chunk_overlap  # 使用用户输入的 chunk_overlap
    )

    loader = TextLoader(tmp_path, encoding='utf-8')
    docs = loader.load()  # 加载文档
    texts = text_splitter.split_documents(docs)  # 拆分文档

    # 根据模型选项选择不同的 OpenAI 嵌入模型
    if model_option == "text-embedding-ada-002(推荐模型，速度快且成本低)":
        embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key, model="text-embedding-ada-002")
    elif model_option == "text-embedding-babbage-001(适用于稍复杂的任务)":
        embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key, model="text-embedding-babbage-001")
    elif model_option == "text-embedding-curie-001(更高质量但更高成本的模型)":
        embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key, model="text-embedding-curie-001")

    # 生成嵌入向量
    text_contents = [t.page_content for t in texts]
    embeddings = embeddings_model.embed_documents(text_contents)

    # 检查嵌入向量是否包含多个特征
    if len(embeddings[0]) < 2:
        raise ValueError("嵌入向量的特征维度不足，无法进行 TSNE 可视化。")

    # 保存结果
    df = pd.DataFrame({
        "text": text_contents,
        "vector": embeddings,
    })

    # 使用完整的 UUID 作为文件名，避免重复
    name_base = f"{uuid.uuid4().hex}"
    return df, name_base
