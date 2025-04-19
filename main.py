import streamlit as st
from pathlib import Path
# Updated imports for LangChain components
from langchain_community.agent_toolkits.sql.base import create_sql_agent # New
from langchain_community.utilities import SQLDatabase # New
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler # New
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit # New
# Other imports remain the same
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import pandas as pd
import re

load_dotenv()

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use SQLLite 3 Database- Student.db", "Connect to you MySQL Database"]

selected_opt = st.sidebar.radio(
    label="Choose the DB which you want to chat", options=radio_opt
)

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide MySQL Host")
    mysql_user = st.sidebar.text_input("MYSQL User")
    mysql_password = st.sidebar.text_input("MYSQL password", type="password")
    mysql_db = st.sidebar.text_input("MySQL database")
else:
    db_uri = LOCALDB

api_key = os.getenv("GROQ_API_KEY")

if not db_uri:
    st.info("Please enter the database information and uri")

if not api_key:
    st.info("Please add the groq api key")

## LLM model
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
    streaming=True,
)


@st.cache_resource(ttl="2h")
def configure_db(
    db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None
):
    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent / "student.db").absolute()
        print(dbfilepath)
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        # Ensure you have mysql-connector-python installed: pip install mysql-connector-python
        return SQLDatabase(
            create_engine(
                f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
            )
        )


if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)

## toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
)


if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")


def format_table_output(response):
    # Check if response contains tabular data (simple check)
    # More robust checks might be needed depending on LLM output variations
    if isinstance(response, str) and any(
        header in response for header in ["NAME", "CLASS", "SECTION", "MARKS"]
    ):
        try:
            # Attempt to parse space-delimited tables first
            lines = response.strip().split("\n")
            header_line_index = -1
            for i, line in enumerate(lines):
                if all(
                    header in line for header in ["NAME", "CLASS", "SECTION", "MARKS"]
                ):
                    header_line_index = i
                    break

            if header_line_index != -1:
                headers = lines[header_line_index].split()
                data = []
                for line in lines[header_line_index + 1 :]:
                    if line.strip():
                        parts = line.split()
                        # Basic handling for multi-word class names like "Data Science"
                        # This might need refinement based on actual data patterns
                        row_data = {}
                        current_part_index = 0
                        for header in headers:
                            if header == "CLASS" and current_part_index + 1 < len(
                                parts
                            ):
                                # Check if the next part looks like a section (e.g., A, B)
                                if parts[current_part_index + 1] in ["A", "B", "C"]:
                                    row_data[header] = parts[current_part_index]
                                    current_part_index += 1
                                else: # Assume multi-word class
                                    row_data[header] = f"{parts[current_part_index]} {parts[current_part_index+1]}"
                                    current_part_index += 2
                            elif current_part_index < len(parts):
                                row_data[header] = parts[current_part_index]
                                current_part_index += 1
                            else:
                                row_data[header] = None # Handle missing data

                        if len(row_data) == len(headers):
                             data.append(list(row_data.values())) # Ensure order matches headers


                if data: # Only proceed if data was extracted
                    df = pd.DataFrame(data, columns=headers)
                    # Extract text before/after the table if necessary
                    pre_table_text = "\n".join(lines[:header_line_index]).strip()
                    post_table_text = "\n".join(lines[header_line_index + len(data) + 1 :]).strip()

                    if pre_table_text:
                        st.write(pre_table_text)
                    st.dataframe(df)
                    if post_table_text:
                        st.write(post_table_text)
                    return None # Indicate display handled

            # Fallback for Markdown tables if space-delimited parsing failed or wasn't applicable
            if "|" in response:
                tables = []
                # Regex to find markdown tables
                md_table_pattern = r"^\s*\|.*\|\s*\n\s*\|[-| :]*\|\s*\n(\s*\|.*\|\s*\n?)+"
                for match in re.finditer(md_table_pattern, response, re.MULTILINE):
                    table_text = match.group(0)
                    try:
                        # Use StringIO to read the markdown table string as a file
                        # Skip the header separator line (line 2)
                        lines = table_text.strip().split('\n')
                        csv_text = "\n".join([lines[0]] + lines[2:])
                        df = pd.read_csv(pd.StringIO(csv_text), sep='|', skipinitialspace=True)
                        # Clean up empty columns from leading/trailing pipes
                        df = df.iloc[:, 1:-1]
                        df.columns = df.columns.str.strip()
                        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
                        tables.append(df)
                    except Exception as e_md:
                        print(f"Markdown parsing error: {e_md}") # Log error
                        pass # Continue if one table fails

                if tables:
                    # Attempt to remove the raw table text from the response
                    result_text = re.sub(md_table_pattern, '', response, flags=re.MULTILINE).strip()
                    if result_text:
                         st.write(result_text)
                    for df in tables:
                        st.dataframe(df)
                    return None # Indicate display handled

        except Exception as e:
            print(f"Error formatting table output: {e}") # Log error
            # If any parsing fails, fall back to showing the raw response
            return response

    # If not detected as table or parsing failed, return original response
    return response


# Add this function before your agent interaction code
def handle_user_input(query, agent, db, callbacks):
    # Detect simple greetings
    greetings = ["hi", "hello", "hey", "greetings", "howdy"]
    if query.lower().strip() in greetings:
        try:
            tables = db.get_usable_table_names()
            return f"Hello! I'm ready to help you query your database. You can ask me about the following tables: {', '.join(tables)}. What would you like to know?"
        except Exception as e:
            print(f"Error getting table names: {e}")
            return "Hello! How can I help you with the database?"
    else:
        # For actual DB queries, use the agent
        try:
            return agent.run(query, callbacks=callbacks)
        except Exception as e:
            print(f"Agent execution error: {e}")
            return "Sorry, I encountered an error trying to process your request."


if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = handle_user_input(user_query, agent, db, [streamlit_callback])

        # Format the response if it contains table data
        formatted_response = format_table_output(response)
        if formatted_response is not None:
            # If format_table_output handled the display (returned None),
            # we still need to store the original response in history.
            # If it returned the original response, display it normally.
             if formatted_response == response: # Check if it returned the original string
                 st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response}) # Store original response
