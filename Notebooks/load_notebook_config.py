import os
from dotenv import load_dotenv
import yaml
from pyprojroot import here
from langchain_openai import ChatOpenAI
load_dotenv()

with open(here("configs/config.yml")) as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)


class LoadDirectoriesConfig:
    def __init__(self) -> None:
        # Databases directories
        self.local_file = here(app_config["directories"]["local_file"])
        self.backup_file = here(app_config["directories"]["backup_file"])
        # Databases URLs
        self.swiss_faq_url = app_config["urls"]["swiss_faq_url"]
        self.travel_db_url = app_config["urls"]["travel_db_url"]


class LoadOpenAIConfig:
    def __init__(self) -> None:
        os.environ['OPENAI_API_KEY'] = os.getenv("OPEN_AI_API_KEY")
        self.llm = ChatOpenAI(model=app_config["openai_models"]["model"])
        self.embedding_model = app_config["openai_models"]["embedding_model"]


class LoadRAGConfig:
    def __init__(self) -> None:
        self.k = app_config["RAG"]["k"]


class LoadConfig:
    def __init__(self) -> None:
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "Brook AI"
