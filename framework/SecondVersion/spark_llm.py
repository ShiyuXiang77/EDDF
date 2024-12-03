from langchain_community.llms import SparkLLM
from config import SPARK_APP_ID, SPARK_API_KEY, SPARK_API_SECRET

def gen_spark_params(model):
    '''
    构造星火模型请求参数
    '''
    spark_url_tpl = "wss://spark-api.xf-yun.com/{}/chat"
    model_params_dict = {
        "v3.5": {
            "domain": "generalv3.5",
            "spark_url": spark_url_tpl.format("v3.5")
        }
    }
    return model_params_dict[model]

def initialize_spark_llm(model="v3.5"):
    '''
    初始化 Spark LLM
    '''
    spark_api_url = gen_spark_params(model)["spark_url"]
    return SparkLLM(
        spark_api_url=spark_api_url,
        spark_api_id=SPARK_APP_ID,
        spark_api_key=SPARK_API_KEY,
        spark_api_secret=SPARK_API_SECRET,
        streaming=False,
        request_timeout=60
    )
