# Streamlit 版本的工具函数，用于展示文件列表并选择上传的文件名
import os
import streamlit as st
import ast

# 显示某目录下所有特定类型的文件，并返回 Streamlit 的选择框
def show_file_selector(path, title, file_ext):
    try:
        files = sorted([f for f in os.listdir(path) if f.lower().endswith(f'.{file_ext}')])
    except FileNotFoundError:
        st.error("❌ 文件夹路径无效。")
        return None

    if not files:
        st.warning("⚠️ 当前目录下没有可用文件。")
        return None

    selected_file = st.selectbox(f"{title}：", files)
    return os.path.join(path, selected_file)


# 显示某目录下所有文件夹，并提供选择功能
def show_folder_selector(path, title):
    try:
        folders = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
    except FileNotFoundError:
        st.error("❌ 路径不存在")
        return None

    if not folders:
        st.warning("⚠️ 当前路径下没有文件夹。")
        return None

    selected_folder = st.selectbox(f"{title}：", folders)
    return os.path.join(path, selected_folder)



def parse_vector_str(x):
    try:
        # 如果已经是列表就直接返回
        if isinstance(x, list):
            return x
        # 如果是字符串，则安全解析
        return ast.literal_eval(x)
    except Exception:
        return []




def print_all_filenames_with_index(path,work,file_class):
    print(f'———————————————————————{work}———————————————————————')
    try:
        # 获取目录下的所有文件
        files = os.listdir(path)
    except FileNotFoundError:
        print("❌ 文件夹路径无效。")
    # 过滤出文件并排序（按文件名）
    sorted_files = sorted([f for f in files if f.lower().endswith(f'.{file_class}')])
    # 构建带序号的字典（从1开始）
    indexed_dic = {i + 1: name for i, name in enumerate(sorted_files)}
    print("序列号\t文件名")
    for i, j in indexed_dic.items():
        print(str(i) + ':' + j)
    return indexed_dic

def input_index(index_dic):
    product_index = input("(此项可不输入，默认从 首项 开始)\n请输入 序列号：").strip()
    try:
        product = index_dic[int(product_index)]
    except ValueError:
        product = index_dic[1]
        print("\nℹ️ 未输入 序列号，默认从首个开始。")
    except KeyError:
        print("\n❌ 未存在 该序列号。")
        exit()
    return product
