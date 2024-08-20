"""
In this module, we define the (fetch_user_flight_information) tool to let the agent see the current user's flight information. Then define tools to search for flights and manage the passenger's bookings stored in the SQL database.

We use ensure_config to pass in the passenger_id in via configurable parameters. The LLM never has to provide these explicitly, they are provided for a given invocation of the graph so that each user cannot access other passengers' booking information.
"""
import sqlite3
from datetime import date, datetime
from typing import Optional
import pytz
from langchain_core.runnables import ensure_config
from langchain_core.tools import tool
from load_config import LoadConfig

CFG = LoadConfig()
# database will be used throughout this module in flight methods
db = CFG.local_file


@tool
def fetch_user_flight_information() -> list[dict]:
    """Fetch all tickets for the user along with corresponding flight information and seat assignments.

    Returns:
        A list of dictionaries where each dictionary contains the ticket details,
        associated flight details, and the seat assignments for each ticket belonging to the user.
    """
    config = ensure_config()  # Fetch from the context
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = """
    SELECT 
        t.ticket_no, t.book_ref,
        f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, f.scheduled_departure, f.scheduled_arrival,
        bp.seat_no, tf.fare_conditions
    FROM 
        tickets t
        JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
        JOIN flights f ON tf.flight_id = f.flight_id
        JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
    WHERE 
        t.passenger_id = ?
    """
    cursor.execute(query, (passenger_id,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results


"""
fetch_user_flight_information full description:
-----------------------------------------------
    Fetches all tickets for the user along with corresponding flight information and seat assignments.

    This function retrieves ticket details, associated flight details, and seat assignments for all
    tickets belonging to the currently configured user. It connects to a SQLite database, queries
    the relevant tables, and returns the data as a list of dictionaries.

    Returns:
        list[dict]: A list of dictionaries where each dictionary contains:
            - ticket_no: The ticket number.
            - book_ref: The booking reference.
            - flight_id: The flight identifier.
            - flight_no: The flight number.
            - departure_airport: The departure airport code.
            - arrival_airport: The arrival airport code.
            - scheduled_departure: The scheduled departure time of the flight.
            - scheduled_arrival: The scheduled arrival time of the flight.
            - seat_no: The seat number assigned to the ticket.
            - fare_conditions: The fare conditions for the ticket.

    Raises:
        ValueError: If no passenger ID is configured in the context.
    """


@tool
def search_flights(
    departure_airport: Optional[str] = None,
    arrival_airport: Optional[str] = None,
    start_time: Optional[date | datetime] = None,
    end_time: Optional[date | datetime] = None,
    limit: int = 20,
) -> list[dict]:
    """Search for flights based on departure airport, arrival airport, and departure time range."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM flights WHERE 1 = 1"
    params = []

    if departure_airport:
        query += " AND departure_airport = ?"
        params.append(departure_airport)

    if arrival_airport:
        query += " AND arrival_airport = ?"
        params.append(arrival_airport)

    if start_time:
        query += " AND scheduled_departure >= ?"
        params.append(start_time)

    if end_time:
        query += " AND scheduled_departure <= ?"
        params.append(end_time)
    query += " LIMIT ?"
    params.append(limit)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results


"""
search_flights full description:
---------------------------------
    Searches for flights based on departure airport, arrival airport, and departure time range.

    This function queries the SQLite database for flights that match the specified criteria and
    returns a list of flights sorted according to the provided filters.

    Args:
        departure_airport (Optional[str]): The airport code from which the flight is departing.
        arrival_airport (Optional[str]): The airport code where the flight is arriving.
        start_time (Optional[date | datetime]): The earliest departure time to filter flights.
        end_time (Optional[date | datetime]): The latest departure time to filter flights.
        limit (int): The maximum number of flights to return. Defaults to 20.

    Returns:
        list[dict]: A list of dictionaries where each dictionary contains the details of a flight
            including all columns from the `flights` table.

    Raises:
        sqlite3.DatabaseError: If there is an issue with reading from the SQLite database.
    """


@tool
def update_ticket_to_new_flight(ticket_no: str, new_flight_id: int) -> str:
    """Update the user's ticket to a new valid flight."""
    config = ensure_config()
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT departure_airport, arrival_airport, scheduled_departure FROM flights WHERE flight_id = ?",
        (new_flight_id,),
    )
    new_flight = cursor.fetchone()
    if not new_flight:
        cursor.close()
        conn.close()
        return "Invalid new flight ID provided."
    column_names = [column[0] for column in cursor.description]
    new_flight_dict = dict(zip(column_names, new_flight))
    timezone = pytz.timezone("Etc/GMT-3")
    current_time = datetime.now(tz=timezone)
    departure_time = datetime.strptime(
        new_flight_dict["scheduled_departure"], "%Y-%m-%d %H:%M:%S.%f%z"
    )
    time_until = (departure_time - current_time).total_seconds()
    if time_until < (3 * 3600):
        return f"Not permitted to reschedule to a flight that is less than 3 hours from the current time. Selected flight is at {departure_time}."

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (
            ticket_no,)
    )
    current_flight = cursor.fetchone()
    if not current_flight:
        cursor.close()
        conn.close()
        return "No existing ticket found for the given ticket number."

    # Check the signed-in user actually has this ticket
    cursor.execute(
        "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"

    # In a real application, you'd likely add additional checks here to enforce business logic,
    # like "does the new departure airport match the current ticket", etc.
    # While it's best to try to be *proactive* in 'type-hinting' policies to the LLM
    # it's inevitably going to get things wrong, so you **also** need to ensure your
    # API enforces valid behavior
    cursor.execute(
        "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?",
        (new_flight_id, ticket_no),
    )
    conn.commit()

    cursor.close()
    conn.close()
    return "Ticket successfully updated to new flight."


"""
update_ticket_to_new_flight full description:
----------------------------------------------
    Updates the user's ticket to a new valid flight.

    This function updates the flight associated with a specific ticket number to a new flight ID.
    It ensures that the new flight ID is valid and that the current user owns the ticket. It also
    checks that the new flight's departure time is at least 3 hours from the current time.

    Args:
        ticket_no (str): The ticket number to be updated.
        new_flight_id (int): The ID of the new flight to which the ticket should be updated.

    Returns:
        str: A message indicating whether the ticket was successfully updated or if there were errors
            such as an invalid flight ID, the flight being too soon, or the user not owning the ticket.

    Raises:
        ValueError: If no passenger ID is configured in the context.
        sqlite3.DatabaseError: If there is an issue with reading or updating the SQLite database.
"""


@tool
def cancel_ticket(ticket_no: str) -> str:
    """Cancel the user's ticket and remove it from the database."""
    config = ensure_config()
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (
            ticket_no,)
    )
    existing_ticket = cursor.fetchone()
    if not existing_ticket:
        cursor.close()
        conn.close()
        return "No existing ticket found for the given ticket number."

    # Check the signed-in user actually has this ticket
    cursor.execute(
        "SELECT flight_id FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"

    cursor.execute(
        "DELETE FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
    conn.commit()

    cursor.close()
    conn.close()
    return "Ticket successfully cancelled."


"""
Cancel_ticket full description:
--------------------------------
    Cancels the user's ticket and removes it from the database.

    This function removes a specified ticket from the database, including its associations with
    flights and other related tables. It ensures that the current user owns the ticket before performing
    the cancellation.

    Args:
        ticket_no (str): The ticket number to be cancelled.

    Returns:
        str: A message indicating whether the ticket was successfully cancelled or if there were errors
            such as no existing ticket being found or the user not owning the ticket.

    Raises:
        ValueError: If no passenger ID is configured in the context.
        sqlite3.DatabaseError: If there is an issue with reading or deleting from the SQLite database.
"""
