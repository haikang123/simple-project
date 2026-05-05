# 1. 核心依赖（LangGraph 状态图 + 类型注解）
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 2. 基础LLM配置
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatTongyi

# 加载 .env 配置
load_dotenv()
# 从环境变量读取通义千问 API 密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 初始化通义千问模型（替换原来的 llm_longcat）
llm_qwen = ChatTongyi(
    model_name="qwen-plus",  # 可按需改为 qwen-turbo / qwen-max
    temperature=0.3,
    dashscope_api_key=DASHSCOPE_API_KEY
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "{context}"),
    ("placeholder", "{messages}")
])
chain = prompt | llm_qwen | StrOutputParser()

# 全局变量
system_prompt= "你是专业Python代码助手，调用代码工具以结构化输出，包含前缀、导入、可运行代码"
max_iterations = 3
flag = "reflect"

# 3. 定义图状态
class GraphState(TypedDict):
    messages: list
    generation: any
    iterations: int
    error: str

# 4. 定义节点1：生成代码
def generate(state:GraphState):
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]

    if error == "yes":
        messages += [
            ("user", "现在，再尝试一次。调用代码工具以结构化输出，包括前缀，导入和代码块。")
        ]

    response = chain.invoke({
        "context": system_prompt,
        "messages": messages
    })

    messages += [("assistant", response)]
    iterations += 1

    return {
        "generation": response,
        "messages": messages,
        "iterations": iterations
    }

# 5. 定义节点2：检查代码
def code_check(state: GraphState):
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    try:
        imports = ""
        if "import" in code_solution:
            imports = code_solution.split("import")[1].split("\n")[0]
            imports = "import " + imports
        code = code_solution

    except:
        imports = ""
        code = code_solution

    try:
        exec(imports)
    except Exception as e:
        print("代码导入失败")
        messages += [("user", f"你的解决方案导入测试失败：{e}")]
        return{
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    try:
        exec(imports + "\n" + code)
    except Exception as e:
        print("代码块检查：失败")
        messages += [("user", f"你的解决方案代码执行失败：{e}")]
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }

# 6. 定义节点3：反思错误
def reflect(state: GraphState):
    messages = state["messages"]
    iterations = state["iterations"]
    code_solution = state["generation"]

    reflections = chain.invoke({
        "context": "分析代码错误原因，给出修正建议",
        "messages": messages
    })

    messages += [("assistant", f"这里是对错误的反思：{reflections}")]
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations
    }

# 7. 定义条件边 决定是否结束流程的函数
def decide_to_finish(state: GraphState):
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == max_iterations:
        print("决策：完成")
        return "end"
    else:
        print("决策：重新解决方案")
        if flag == "reflect":
            return "reflect"
        else:
            return "generate"

# 8. 构建流程
builder = StateGraph(GraphState)
builder.add_node("gen_code", generate)
builder.add_node("check_code", code_check)
builder.add_node("reflect_code", reflect)

builder.add_edge(START, "gen_code")
builder.add_edge("gen_code", "check_code")
builder.add_edge("reflect_code", "gen_code")

builder.add_conditional_edges(
    "check_code",
    decide_to_finish,
    {
        "end": END,
        "reflect": "reflect_code",
        "generate": "gen_code",
    }
)
graph = builder.compile()

# 9. 运行
if __name__ == "__main__":
    initial_state = {
        "messages": [("user", "生成一个查看当前时间日期的代码精确到分钟即可，生成本地运行")],
        "generation": None,
        "iterations": 0,
        "error": "no"
    }

    result = graph.invoke(initial_state)
    print("\n===== 最终生成结果 =====")
    print(result["generation"])