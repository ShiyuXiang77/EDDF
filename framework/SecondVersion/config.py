from dotenv import load_dotenv, find_dotenv
import os

# 加载环境变量
_ = load_dotenv(find_dotenv())
# 设置环境变量 API_KEY
os.environ["IFLYTEK_SPARK_APP_ID"] = 'a54ddac8'
os.environ["IFLYTEK_SPARK_API_KEY"] = 'a9cadac070a5ec1ccd54a110497e2686'
os.environ["IFLYTEK_SPARK_API_SECRET"] = 'YjM2ZmE5NDcyNjA0MjIyNGViNDI5NTI0'

SPARK_APP_ID = os.environ["IFLYTEK_SPARK_APP_ID"]
SPARK_API_KEY = os.environ["IFLYTEK_SPARK_API_KEY"]
SPARK_API_SECRET = os.environ["IFLYTEK_SPARK_API_SECRET"]