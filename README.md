#  一些简单的项目
基于langchain学习过程中的项目

## 文件说明

### 1. RAG
```
一个简单rag搭建，包括加载文件，文件分割，文档嵌入，向量储存。
```

### 2. Agent
```
一个简单的用于代码生成的助手agent，使用LangGraph进行搭建。
```

### 3. 长期记忆
```
一个简单的记忆模板，使对话记录保存在你的本地硬盘，进行记忆的测试。
```

##  环境准备
- Python 3.10+
- 阿里云通义千问 API 密钥（获取地址：https://dashscope.console.aliyun.com/）

---

##  快速启动

### 1. 克隆项目
```
git clone https://github.com/haikang123/AI-agent-demo.git
cd AI-agent-demo
```

### 2. 安装依赖
```
pip install -r requirements.txt
```


### 3. 配置说明
在项目根目录新建 .env 文件，填写：
```
DASHSCOPE_API_KEY="你的API密钥"
```
