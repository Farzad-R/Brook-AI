import uuid
import shutil
from agentic_system_design.construct_graph import BrookAIGraph
from load_config import LoadDirectoriesConfig
from utils.utilities import _print_event
from typing import List, Tuple
from download_data import download_travel_data
import os

CFG_DIRECTORIES = LoadDirectoriesConfig()
db = CFG_DIRECTORIES.local_file
backup_file = CFG_DIRECTORIES.backup_file

# Check if the database and backup files exist
db_exists = os.path.exists(db)
backup_file_exists = os.path.exists(backup_file)

# If either file does not exist, call the download function
if not db_exists or not backup_file_exists:
    download_travel_data()

graph_instance = BrookAIGraph()
graph = graph_instance.Compile_graph()

shutil.copy(backup_file, db)
thread_id = str(uuid.uuid4())
print("@@@@@@@@@@@@@@@@@@@@@")
print("thread_id:", thread_id)
print("@@@@@@@@@@@@@@@@@@@@@")

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "passenger_id": "3442 587242",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
        "recursion_limit": 50
    }
}


class ChatBot:
    @staticmethod
    def respond(chatbot: List, message: str) -> Tuple:
        _printed = set()
        events = graph.stream(
            {"messages": ("user", message)}, config, stream_mode="values"
        )
        for event in events:
            _print_event(event, _printed)
        # print("************************************")
        # print(event)
        # print("************************************")
        snapshot = graph.get_state(config)
        while snapshot.next:
            result = graph.invoke(
                None,
                config,
            )
            snapshot = graph.get_state(config)
        chatbot.append(
            (message, snapshot[0]["messages"][-1].content))
        return "", chatbot, None
