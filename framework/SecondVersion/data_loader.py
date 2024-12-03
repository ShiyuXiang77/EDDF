from langchain_community.document_loaders import PyPDFLoader
import json
import os

def fetch_content_from_url(url):
    '''
    :param url: https://arxiv.org/pdf/*,注意是pdf不是abs，abs是html文件
    :return: 整篇论文内容
    '''
    # 下载PDF文件
    pdf_loader = PyPDFLoader(file_path=url)
    pages = pdf_loader.load()
    # 合并所有页内容为一个字符串
    content = " ".join([page.page_content for page in pages])
    # 查找"References"
    reference_start = content.find("References")
    # 查找"Preliminaries"
    preliminaries_start = content.find("Methodology")
    if preliminaries_start == -1:  # 如果没有找到 "Methodology"
        preliminaries_start = content.find("Method")  # 尝试查找 "Method"
    if reference_start != -1:
        content = content[:reference_start]
    if preliminaries_start != -1:
        content = content[preliminaries_start:]
    return content

def append_to_json(file_path, data):
    '''
    将数据追加到 JSON 文件
    '''
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([], f)

    with open(file_path, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)

    existing_data.append(data)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

def load_attack_knowledge(file_path):
    '''
    加载攻击知识 JSON 文件
    '''
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
