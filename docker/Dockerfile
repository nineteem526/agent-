# Dockerfile — 把 Agent 打包成容器
# ================================
# 构建：docker build -t agent-app -f docker/Dockerfile .
# 运行：docker run -p 8501:8501 --env-file .env agent-app

FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 创建数据目录
RUN mkdir -p /app/data/chroma_db

# Streamlit 端口
EXPOSE 8501

# 启动
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0"]
