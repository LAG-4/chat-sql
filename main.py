import os
import re
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine
import logging
from langchain.agents.agent_types import AgentType
from langchain.agents import AgentExecutor
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq

#
# Page setup
#
load_dotenv()
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

#
# Sidebar: choose SQLite vs MySQL
#
LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"
radio_opt = ["Use SQLLite 3 Database- Student.db",
             "Connect to your MySQL Database"]
selected_opt = st.sidebar.radio("Choose the DB to chat with", radio_opt)

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password",
                                           type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_uri = LOCALDB

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.sidebar.error("Please set GROQ_API_KEY in your .env")

#
# Configure DB connection
#
@st.cache_resource(ttl="2h")
def configure_db(uri, host=None, user=None, password=None, dbname=None):
    if uri == LOCALDB:
        dbfile = (Path(__file__).parent / "student.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True)
        engine = create_engine("sqlite:///", creator=creator)
        return SQLDatabase(engine)
    else:
        if not (host and user and password and dbname):
            st.sidebar.error("Provide all MySQL details")
            st.stop()
        engine = create_engine(
            f"mysql+mysqlconnector://{user}:{password}@"
            f"{host}/{dbname}"
        )
        return SQLDatabase(engine)

db = configure_db(db_uri, mysql_host, mysql_user,
                  mysql_password, mysql_db) if db_uri == MYSQL \
    else configure_db(db_uri)

#
# LLM and Agent setup
#
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
    streaming=True,
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    max_iterations=10,         # raise iteration limit
    max_execution_time=60      # raise time limit (seconds)
)
# wrap into an executor so you can also control iteration here if needed
executor = AgentExecutor.from_agent_and_tools(
    agent=agent.agent, tools=agent.tools, max_iterations=10
)

#
# Utility: direct SQL for average & mode
#
def get_avg_and_mode(class_name: str):
    avg_sql = """
        SELECT ROUND(AVG(marks),2) AS average_marks
        FROM student
        WHERE class = :cls
    """
    mode_sql = """
        SELECT marks AS mode_marks
        FROM (
          SELECT marks, COUNT(*) AS cnt
          FROM student
          WHERE class = :cls
          GROUP BY marks
        ) t
        ORDER BY cnt DESC
        LIMIT 1
    """
    avg_df = pd.read_sql(avg_sql, db.engine, params={"cls": class_name})
    mode_df = pd.read_sql(mode_sql, db.engine, params={"cls": class_name})
    return float(avg_df["average_marks"].iloc[0]), int(mode_df["mode_marks"].iloc[0])

#
# Message history
#
if "messages" not in st.session_state \
   or st.sidebar.button("Clear message history"):
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input("Ask anything from the database")

#
# Table formatter (unchanged from your code)
#
def format_table_output(response):
    # ... your existing format_table_output code here ...
    return response  # placeholder

#
# Main handler
#
def handle_user_input(query: str):
    q = query.lower()
    # direct‐SQL path for "average" or "mode" questions
    avg_mode_match = re.search(
        r"(average|mode).+class\s+(\w+)", q, re.IGNORECASE
    )
    if avg_mode_match:
        what, cls = avg_mode_match.groups()
        avg, mode = get_avg_and_mode(cls)
        if "average" in what:
            return f"Class {cls} → average marks: {avg:.2f}"
        if "mode" in what:
            return f"Class {cls} → mode marks: {mode}"
    # otherwise fall back to agent
    try:
        cb = StreamlitCallbackHandler(st.container())
        return executor.run(query, callbacks=[cb])
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Agent error", exc_info=True)
        return "Sorry, I couldn't process that."

#
# When the user submits a query
#
if user_query:
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        resp = handle_user_input(user_query)
        formatted = format_table_output(resp)
        if formatted is not None:
            st.write(formatted)

    st.session_state.messages.append(
        {"role": "assistant", "content": resp}
    )
