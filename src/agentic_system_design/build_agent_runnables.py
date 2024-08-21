"""
Here, we will create an assistant for every workflow.

1. Flight booking assistant
2. Hotel booking assistant
3. Car rental assistant
4. Excursion assistant
5. and finally, a "primary assistant" to route between these

Below, define the Runnable objects to power each assistant. Each Runnable has a prompt, LLM, and schemas for the tools scoped to that assistant. Each specialized / delegated assistant additionally can call the CompleteOrEscalate tool to indicate that the control flow should be passed back to the primary assistant. This happens if it has successfully completed its work or if the user has changed their mind or needs assistance on something that beyond the scope of that particular workflow.

"""


from tools.company_lookup_policy_tools import lookup_policy
from tools.flights_tools import search_flights, update_ticket_to_new_flight, cancel_ticket
from tools.hotels_tools import search_hotels, book_hotel, update_hotel, cancel_hotel
from tools.excursions_tools import search_trip_recommendations, book_excursion, update_excursion, cancel_excursion
from tools.car_rental_tools import search_car_rentals, book_car_rental, update_car_rental, cancel_car_rental
from load_config import LoadOpenAIConfig, LoadRAGConfig
from langchain_community.tools.tavily_search import TavilySearchResults
from agentic_system_design.build_agent_prompts import AgentPrompts
from agentic_system_design.complete_or_escalate import CompleteOrEscalate
from agentic_system_design.build_agent_assistants import ToFlightBookingAssistant, ToBookCarRentalAssistant, ToHotelBookingAssistant, ToBookExcursionAssistant

AGENT_PROMPTS = AgentPrompts()
CFG_OPENAI = LoadOpenAIConfig()
CFG_RAG = LoadRAGConfig()


class BrookAIAgentRunnables:
    def __init__(self) -> None:
        self.update_flight_safe_tools, self.update_flight_sensitive_tools, self.update_flight_runnable = self.build_update_flight_runnable()
        self.book_hotel_safe_tools, self.book_hotel_sensitive_tools, self.book_hotel_runnable = self.build_book_hotel_runnable()
        self.book_car_rental_safe_tools, self.book_car_rental_sensitive_tools, self.book_car_rental_runnable = self.build_book_car_rental_runnable()
        self.book_excursion_safe_tools, self.book_excursion_sensitive_tools, self.book_excursion_runnable = self.build_book_excursion_runnable()
        self.primary_assistant_tools, self.primary_assistant_runnable = self.build_primary_assistant_runnable()

    def build_update_flight_runnable(self):
        update_flight_safe_tools = [search_flights]
        update_flight_sensitive_tools = [
            update_ticket_to_new_flight, cancel_ticket]
        update_flight_tools = update_flight_safe_tools + update_flight_sensitive_tools
        update_flight_runnable = AGENT_PROMPTS.flight_booking_prompt | CFG_OPENAI.llm.bind_tools(
            update_flight_tools + [CompleteOrEscalate]
        )
        return update_flight_safe_tools, update_flight_sensitive_tools, update_flight_runnable

    def build_book_hotel_runnable(self):
        book_hotel_safe_tools = [search_hotels]
        book_hotel_sensitive_tools = [book_hotel, update_hotel, cancel_hotel]
        book_hotel_tools = book_hotel_safe_tools + book_hotel_sensitive_tools
        book_hotel_runnable = AGENT_PROMPTS.book_hotel_prompt | CFG_OPENAI.llm.bind_tools(
            book_hotel_tools + [CompleteOrEscalate]
        )
        return book_hotel_safe_tools, book_hotel_sensitive_tools, book_hotel_runnable

    def build_book_car_rental_runnable(self):
        book_car_rental_safe_tools = [search_car_rentals]
        book_car_rental_sensitive_tools = [
            book_car_rental,
            update_car_rental,
            cancel_car_rental,
        ]
        book_car_rental_tools = book_car_rental_safe_tools + book_car_rental_sensitive_tools
        book_car_rental_runnable = AGENT_PROMPTS.book_car_rental_prompt | CFG_OPENAI.llm.bind_tools(
            book_car_rental_tools + [CompleteOrEscalate]
        )
        return book_car_rental_safe_tools, book_car_rental_sensitive_tools, book_car_rental_runnable

    def build_book_excursion_runnable(self):
        book_excursion_safe_tools = [search_trip_recommendations]
        book_excursion_sensitive_tools = [
            book_excursion, update_excursion, cancel_excursion]
        book_excursion_tools = book_excursion_safe_tools + book_excursion_sensitive_tools
        book_excursion_runnable = AGENT_PROMPTS.book_excursion_prompt | CFG_OPENAI.llm.bind_tools(
            book_excursion_tools + [CompleteOrEscalate]
        )
        return book_excursion_safe_tools, book_excursion_sensitive_tools, book_excursion_runnable

    def build_primary_assistant_runnable(self):
        primary_assistant_tools = [
            TavilySearchResults(max_results=CFG_RAG.tavily_search_max_results),
            search_flights,
            lookup_policy,
        ]
        primary_assistant_runnable = AGENT_PROMPTS.primary_assistant_prompt | CFG_OPENAI.llm.bind_tools(
            primary_assistant_tools
            + [
                ToFlightBookingAssistant,
                ToBookCarRentalAssistant,
                ToHotelBookingAssistant,
                ToBookExcursionAssistant,
            ]
        )
        return primary_assistant_tools, primary_assistant_runnable
