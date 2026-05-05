import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatTongyi
from dotenv import load_dotenv
# 加载 .env 配置
load_dotenv()
# 从环境变量读取通义千问 API 密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# 初始化通义千问模型
llm_qwen = ChatTongyi(
    model_name="qwen-plus",  # 可按需改为 qwen-turbo / qwen-max
    temperature=0.3,
    dashscope_api_key=DASHSCOPE_API_KEY
)
#1 加载文档
docs_path = Path("./docs")
docs_path.mkdir(exist_ok=True)

documents = []
for file_path in docs_path.glob("*.*"):
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
        elif suffix in [".docx", ".doc"]:
            loader = Docx2txtLoader(str(file_path))
            docs = loader.load()
        else:
            continue

        for doc in docs:
            doc.metadata["source"] = str(file_path)
        documents.extend(docs)
    except Exception as e:
        print(f"文件 {file_path} 加载失败：{e}")

if not documents:
    print("没有可加载的文档！")
    exit()


#2 文本分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=126,
    separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
)
texts = text_splitter.split_documents(documents)


#3 文档嵌入，并且嵌入缓存
underlying_embeddings = DashScopeEmbeddings()
fs = LocalFileStore("./cache")
embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, fs, namespace=underlying_embeddings.model
)

#4 向量存储
db = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
if not db.get()["ids"]:
    db.add_documents(texts)

retriever_mmr = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10
    }
)


#5 系统提示词模板
prompt_template = """
你是严谨的知识问答助手，只根据提供的资料回答。
规则：
1. 不编造、不扩展资料外内容
2. 数学公式、符号用自然语言清晰描述，不输出乱码与特殊排版
3. 忽略乱码、乱码字符、无效符号
4. 回答简洁、专业、条理清晰

资料：
{context}

问题：{question}
回答：
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


#6 rag链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_qwen,
    chain_type="stuff",
    retriever=retriever_mmr,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)


#7 提问
query = "神经网络是什么？"
result = qa_chain.invoke(query)


#8 输出结果
print("===== 最终回答 =====")
print(result["result"])

#详细输出
# print("\n===== 来源片段 =====")
# docs_with_score = db.similarity_search_with_score(query, k=4)
# for idx, (doc, score) in enumerate(docs_with_score, 1):
#     print(f"\n片段 {idx} | 原始距离分数：{score:.4f}")
#     print("来源：", doc.metadata["source"])
#     print(doc.page_content)