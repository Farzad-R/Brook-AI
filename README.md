# Brook the Booking Agent: TravelEase’s AI Agent

## Overview
Brook is an AI customer service chatbot for TravelEase, utilizing Swiss Airlines' database to assist users with travel-related queries and bookings.

## Technologies
- **Programming Language:** Python
- **Language Models:** OpenAI GPT models
- **Agents Framework:** LangGraph
- **Monitoring System:** LangSmith
- **User Interface:** Gradio
- **Database Interaction:** SQLAlchemy

## Features
- **Customer History:** Automatically fetch historical data
- **Web Search:** Provide additional information via web searches
- **RAG:** Answer inquiries based on company policies
- **Flights:** Search, update, and cancel flight tickets
- **Car Rentals:** Search, book, update, and cancel car rentals
- **Hotels:** Search, book, update, and cancel hotel reservations
- **Excursions:** Search, book, update, and cancel excursions

## Database
For a detailed database description, refer to `Boork AI POC DataBase Report.pdf`.

## System Design
Brook uses a supervisor agentic design with one main LLM orchestrating five specialized assistants. This design supports various tasks through 18 tools (e.g., RAG, web search, and travel planning).

- **Total Agents:** 6 (1 supervisor, 5 assistants)
- **Total Tools:** 18

## Setup Instructions
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Farzad-R/Brook-AI.git
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure API Keys:**
   Edit the `.env` file and add:
   ```
   OPEN_AI_API_KEY=
   TAVILY_API_KEY=
   LANGCHAIN_API_KEY=
   ```

6. **Run the User Interface:**
   ```bash
   python src/app.py
   ```

   Note: This will download and store two SQL databases (approx. 200MB) in the `data` folder.

7. **Run Agents in Notebook:**
   Open `src/brook_ai_notebook` and execute the cells.

8. **Configure Project Settings:**
   Modify `config/config.yml` as needed.

## Documentation
For detailed tool descriptions, the database report, and system design schema, refer to the `documentation` folder.

For additional support and details, please refer to the attached documentation and images in the `brook_ai_poc` folder.