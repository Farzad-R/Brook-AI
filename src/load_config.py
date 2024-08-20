import os
from dotenv import load_dotenv
import yaml
from pyprojroot import here
import shutil
from langchain_openai import ChatOpenAI
load_dotenv()


class LoadConfig:
    def __init__(self) -> None:
        with open(here("configs/config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)
        # Load OpenAI credentials
        self.load_openai_cfg()

        # Databases directories
        self.local_file = app_config["directories"]["local_file"]
        self.backup_file = app_config["directories"]["backup_file"]

        # Databases URLs
        self.swiss_faq_url = app_config["urls"]["swiss_faq_url"]
        self.travel_db_url = app_config["urls"]["travel_db_url"]

        # LLM
        self.llm = ChatOpenAI(model=app_config["llm"]["model"])

    def load_openai_cfg(self):
        os.environ['OPENAI_API_KEY'] = os.getenv("OPEN_AI_API_KEY")

    def create_directory(self, directory_path: str):
        """
        Create a directory if it does not exist.

        Parameters:
            directory_path (str): The path of the directory to be created.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def remove_directory(self, directory_path: str):
        """
        Removes the specified directory.

        Parameters:
            directory_path (str): The path of the directory to be removed.

        Raises:
            OSError: If an error occurs during the directory removal process.

        Returns:
            None
        """
        if os.path.exists(directory_path):
            try:
                shutil.rmtree(directory_path)
                print(
                    f"The directory '{directory_path}' has been successfully removed.")
            except OSError as e:
                print(f"Error: {e}")
        else:
            print(f"The directory '{directory_path}' does not exist.")
