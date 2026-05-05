#最典型的长期记忆管理包括向量存储和知识图谱
#这里我们两种都介绍一下，向量存储是目前最主流的长期记忆管理方式，知识图谱则适合结构化知识的存储和推理。
# @tool 必须要"""  """注释，否则无法被识别为工具函数
"""    基于向量的记忆存储
根据向量数据库的检索，上下文智能匹配最相关的记忆片段。
这种方法不依赖于严格的时间顺序，而是通过语义相关性进行匹配
"""
# 必需的基础库
import os
from dotenv import load_dotenv
import uuid  #uuid 是 Python 自带的库，专门用来生成【全球唯一的 ID】
from typing import List

# LangChain 核心向量存储 & 嵌入模型
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

# LangChain 文档 & 工具
from langchain_core.documents import Document  #存储 “内容” 和 “元数据”，包含 page_content 和 metadata
from langchain_core.tools import tool  #用 @tool 装饰普通的 Python 函数，就能把它变成 LLM 可以识别和调用的 “工具”
from langchain_core.runnables import RunnableConfig #用于传递运行时配置,通过 config 来区分当前是哪个用户在调用，从而实现记忆隔离

# LangGraph 状态 & 流程
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver #它能保存图的 “状态”（也就是对话历史），并在下次运行时恢复
from langgraph.graph import MessagesState
# 消息处理
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# 加载 .env 配置
load_dotenv()
# 从环境变量读取通义千问 API 密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# 大模型（qwen-plus：工具调用能力拉满，适配国产模型）
from langchain_community.chat_models import ChatTongyi
model = ChatTongyi(
    model_name="qwen-plus",
    temperature=0.3,  # 降低温度，提升工具调用稳定性
)


# 初始化【持久化】向量存储（重启不丢失）
embedding = DashScopeEmbeddings()
recall_vector_store = Chroma(
    collection_name="user_long_term_memory",
    embedding_function=embedding,
    persist_directory="./chroma_mem", # 本地持久化路径,设置后，数据会保存在硬盘的 chroma_memory 文件夹下。
)


# 定义提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", """
你是一个具备长期记忆能力的智能助手，严格遵守以下规则：
1. 【记忆检索强制要求】：回答用户任何问题前，必须先调用 search_recall_memory 工具，检索用户的历史长期记忆
2. 【记忆保存强制要求】：用户提到个人信息、姓名、爱好、学习内容、目标、偏好时，必须调用 save_recall_memory 工具，保存核心信息
3. 【回答要求】：必须结合检索到的长期记忆回答，不要生硬复读记忆，自然融合到回答中
4. 【工具调用要求】：严格按照工具定义的参数格式调用，不要编造参数
用户历史记忆：{recall_memory}   
    """),    #{recall_memory}是占位符。后面我们会把从向量库查到的记忆填到这里。
    ("placeholder", "{messages}")  #消息占位符。这里会填入历史的对话记录（HumanMessage, AIMessage 列表
])

#定义对记忆进行存储和检索的核心工具函数
def get_user_id(config:RunnableConfig) -> str:
    """获取用户ID
    Args:
        config: 运行时配置
    Returns:
        str: 用户ID
    Raises:
        ValueError: 如果用户ID未找到
    """
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("需要用户提供ID才能进行记忆保存")
    return user_id

@tool
def save_recall_memory(memory:str , config:RunnableConfig) -> str:
    """保存记忆内容到向量存储
    Args:
        config: 运行时配置
        memory_content: 需要保存的记忆内容
    Returns:
        str: 保存结果
    """
    user_id = get_user_id(config)
    doc = Document(
        page_content=memory,
        id=str(uuid.uuid4()),  #给每一段记忆，生成一个永不重复的唯一编号。
        metadata={"user_id": user_id}
    )
    recall_vector_store.add_documents([doc])
    #recall_vector_store.persist()
    return f" 记忆已保存：{memory}"

@tool
def search_recall_memory(query:str, config:RunnableConfig) -> List[str]:
    """搜索相关记忆内容    
    Args:
        config: 运行时配置
        query: 搜索查询内容
    Returns:
        List[str]: 搜索结果列表
    """
    user_id = get_user_id(config)
    docs = recall_vector_store.similarity_search(
        query, k=3, filter={"user_id": user_id}
    )
    return [doc.page_content for doc in docs] 

# 定义工具列表
tools = [save_recall_memory, search_recall_memory]
model_with_tools = model.bind_tools(tools) #这是最关键的一步。把工具列表 “绑定” 到模型上。

#声明state用于储存与对话相关的记忆
class State(MessagesState):
    recall_memories: List[str]

#【前置节点】对话开始前，自动加载相关长期记忆
def load_memories(state: State, config: RunnableConfig):
    """加载与当前对话相关的记忆
    Args:
        state: 当前状态
        config: 运行时配置
    工作流程：
    1. 获取当前对话内容
    2. 将对话内容截断到800个字符
    3. 基于对话内容搜索相关记忆
    Returns:
        State: 更新后的状态，包含相关记忆
    """
    #获取当前记忆对话内容字符串
    convo_str = "\n".join([msg.content for msg in state["messages"]])
    convo_str = convo_str[:800]
    recall_memories = search_recall_memory.invoke(convo_str, config)
    return {"recall_memories": recall_memories}

#【核心节点】LLM决策生成，调用工具
def agent_node(state:State) :
    """
    Args:
        state: 当前状态
    Returns:
        State: 更新后的状态和代理回应
    """
    bound_prompt = prompt | model_with_tools # 拼接提示词
    recall_str = "\n".join(state["recall_memories"])
    response = bound_prompt.invoke({
        "messages": state["messages"],
        "recall_memory": recall_str
    })
    return {"messages": [response]}

#【路由节点】条件边判断：调用工具还是结束对话
def route_tools(state: State):
    """
    Args:
        state: 当前状态       
    Returns:
        Literal["tools","__end__"] 返回下一步是使用工具还是结束对话
    工作流程：
    1. 获取最后一条信息
    2. 检查是否需要使用工具
    3. 根据检查结果决定路由
    """
    #获取最后一条信息
    last_msg = state["messages"][-1]
    #如果消息需要调用工具，则返回tools
    if last_msg.tool_calls:
        return "tools"
    #否则结束当前对话
    return END

#构建流程图
builder = StateGraph(State)

builder.add_node("load_memories", load_memories)
builder.add_node("agent", agent_node)
builder.add_node("tools",ToolNode(tools)) #工具调用节点

builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_conditional_edges(
    "agent",
    route_tools,
    {"tools": "tools", END: END}
)
builder.add_edge("tools", "agent")

# 编译智能体，启用对话记忆持久化
checkpointer = MemorySaver()  #创建记忆保存器
graph = builder.compile(checkpointer=checkpointer) #编译流程图并启用检查点功能

# 输出控制
def get_stream_chunk(chunk): # 去掉 -> str
    """直接打印"""
    for node, data in chunk.items():
        if "messages" in data:
            msg = data["messages"][-1]
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                continue
            if hasattr(msg, "content") and msg.content:
                print("输出：", msg.content) # 这里直接打印

# 下面循环里直接调用即可，不用赋值

#测试入口
if __name__ == "__main__":
    config = {"configurable": {"user_id": "user1", "thread_id": "thread1"}}

    for chunk in graph.stream(
        {"messages": [HumanMessage(content="我叫小明，喜欢编程，正在学习LangGraph")]},
        config=config
    ):
        get_stream_chunk(chunk)

    for chunk in graph.stream(
        {"messages": [HumanMessage(content="我最近在学什么？")]},
        config=config
    ):
        get_stream_chunk(chunk)

    # for chunk in graph.stream(
    #     {"messages": [HumanMessage(content="重点内容是什么？")]},
    #     config=config
    # ):
    #     get_stream_chunk(chunk)