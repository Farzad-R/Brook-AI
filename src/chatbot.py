import os
import openai
from typing_extensions import TypedDict
from typing import Annotated
from datetime import date, datetime
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.message import AnyMessage, add_messages
# Loading the project's configs
from load_config import LoadConfig
# Loading from RAG folder
from RAG.load_company_policy_docs import load_swiss_faq
from RAG.company_policies_agentic_RAG import VectorStoreRetriever, lookup_policy
# Loading from tools folder
from tools.flights_tools import fetch_user_flight_information, search_flights, update_ticket_to_new_flight, cancel_ticket
from tools.hotels_tools import search_hotels, book_hotel, update_hotel, cancel_hotel
from tools.excursions_tools import search_trip_recommendations, book_excursion, update_excursion, cancel_excursion
from tools.car_rental_tools import search_car_rentals, book_car_rental, update_car_rental, cancel_car_rental
# Loading from utils folder
from utils.download_data import download_data
from utils.utilities import handle_tool_error, create_tool_node_with_fallback


CFG = LoadConfig()

# Download the databases if they do not already exist in the data directory
if not os.path.exists(CFG.local_file) or not os.path.exists(CFG.backup_file):
    download_data(overwrite=False)

docs = load_swiss_faq()
retriever = VectorStoreRetriever.from_docs(docs, openai.Client())


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# from langchain_anthropic import ChatAnthropic
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + \
                    [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


llm = ChatOpenAI(model="gpt-4-turbo-preview")

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n\n{user_info}\n"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

general_tools = [
    TavilySearchResults(max_results=1),
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    update_ticket_to_new_flight,
    cancel_ticket,
    search_car_rentals,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    search_hotels,
    book_hotel,
    update_hotel,
    cancel_hotel,
    search_trip_recommendations,
    book_excursion,
    update_excursion,
    cancel_excursion,
]

sensitive_tools = [
    update_ticket_to_new_flight,
    cancel_ticket,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    book_hotel,
    update_hotel,
    cancel_hotel,
    book_excursion,
    update_excursion,
    cancel_excursion,
]

part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    part_1_tools)
