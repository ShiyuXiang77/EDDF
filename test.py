from langchain_community.document_loaders import PyPDFLoader

def fetch_arxiv_pdf_content(url):
    # 下载 PDF 文件
    pdf_loader = PyPDFLoader(file_path=url)
    pages = pdf_loader.load()
    # 合并所有页内容为一个字符串
    content = " ".join([page.page_content for page in pages])
    #查找"References"
    reference_start = content.find("References")
    #查找"Preliminaries"
    preliminaries_start = content.find("Preliminaries")
    if reference_start:
        content = content[:reference_start]
    if preliminaries_start:
        content = content[preliminaries_start:]
    return content



# 示例：arXiv 论文的 PDF 链接
url = "https://arxiv.org/pdf/2411.11683"
content = fetch_arxiv_pdf_content(url)
print(content)
print(len(content))