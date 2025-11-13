import streamlit as st
import os
import re
import sqlite3
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import logging
import asyncio
from typing import Dict

from langchain.agents.agent_types import AgentType
from langchain.agents import AgentExecutor
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI

from agno.agent import Agent
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.google import Gemini

from google import genai

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="AI Assistant Hub",
    page_icon="✨",
    layout="wide"
)

# ============================================================================
# SIDEBAR - Mode Selection
# ============================================================================
st.sidebar.title("✨ AI Assistant Hub")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Select Mode",
    ["💰 Financial Analysis", "🗄️ SQL Database Chat", "🎭 AI Advisors"]
)

st.sidebar.markdown("---")

# ============================================================================
# FINANCIAL ANALYSIS MODE
# ============================================================================
if app_mode == "💰 Financial Analysis":
    st.title("📈 Quantum Finance")
    st.subheader("AI-Powered Market Analysis")
    st.markdown("---")
    
    # Create a single comprehensive financial agent
    financial_agent = Agent(
        name="Comprehensive Financial Agent",
        model=Gemini(id="gemini-2.5-flash"),
        tools=[YFinanceTools(), DuckDuckGoTools()],
        instructions=[
            "You are a comprehensive financial analysis agent with access to stock data and web search",
            "ALWAYS present ALL data in tabular format when possible",
            "For stock analysis: use YFinance tools to get current price, fundamentals, analyst recommendations",
            "For news and market sentiment: use DuckDuckGo search to find recent news and articles",
            "Present analyst recommendations with consensus ratings in a table",
            "Include target price ranges and average price targets",
            "Structure your output with clear headings and sections",
            "Always cite sources for external information with dates",
            "Provide actionable insights and clear recommendations",
        ],
        markdown=True,
    )
    
    # Tabs
    tab1, tab2 = st.tabs(["📊 Market Analysis", "💬 AI Assistant"])
    
    with tab1:
        st.header("Stock Analysis")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            stock_symbol = st.text_input("Enter Ticker Symbol", value="AAPL", placeholder="E.g., AAPL, MSFT")
            
        with col2:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Complete Analysis", "News Impact", "Quick Overview"]
            )
            
        with col3:
            st.write("")  # spacing
            st.write("")  # spacing
            analyze_button = st.button("🔍 Analyze", use_container_width=True, type="primary")
        
        # Quick stock buttons
        st.write("**Popular stocks:**")
        col1, col2, col3, col4 = st.columns(4)
        
        quick_stocks = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        for col, stock in zip([col1, col2, col3, col4], quick_stocks):
            with col:
                if st.button(stock, use_container_width=True, key=f"quick_{stock}"):
                    stock_symbol = stock
                    analyze_button = True
        
        st.markdown("---")
        
        # Display analysis
        if analyze_button and stock_symbol:
            with st.spinner('🔄 Analyzing market data...'):
                try:
                    if analysis_type == "Complete Analysis":
                        prompt = f"""Provide a comprehensive analysis for {stock_symbol}:
                        1. Current stock price and key metrics
                        2. Fundamental analysis (P/E, market cap, revenue, etc.)
                        3. Analyst recommendations and target prices
                        4. Recent news and market sentiment (search the web)
                        5. Technical indicators if available
                        6. Investment outlook and recommendations
                        
                        Present everything in tables and organized sections."""
                        
                    elif analysis_type == "News Impact":
                        prompt = f"""Find and analyze the latest news for {stock_symbol}:
                        1. Search for recent news articles (last 7 days)
                        2. Summarize key news items in a table
                        3. Assess market impact (Positive/Neutral/Negative)
                        4. Get current stock price
                        5. Provide outlook based on news sentiment"""
                        
                    else:  # Quick Overview
                        prompt = f"""Provide a quick overview for {stock_symbol}:
                        1. Current price and today's change
                        2. Key metrics (P/E, Market Cap)
                        3. Latest analyst recommendation
                        4. One-sentence market sentiment
                        
                        Keep it concise and in table format."""
                    
                    st.subheader(f"📊 Analysis: ${stock_symbol}")
                    
                    response = financial_agent.run(prompt)
                    st.markdown(response.content)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Tip: Make sure you have GEMINI_API_KEY set in your .env file")
    
    with tab2:
        st.header("AI Financial Assistant")
        st.write("Ask questions about markets, stocks, or investment strategies.")
        
        # Example questions
        st.write("**Try asking about:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💡 Investment tips", use_container_width=True):
                st.session_state.preset_question = "What are some investment tips for beginners?"
        with col2:
            if st.button("💰 Market trends", use_container_width=True):
                st.session_state.preset_question = "What are the current trends in the stock market?"
        
        st.markdown("---")
        
        # Chat input
        user_question = st.text_input(
            "Your question:", 
            placeholder="e.g., How do interest rates affect stocks?",
            value=st.session_state.get("preset_question", ""),
            key="finance_question"
        )
        
        if "preset_question" in st.session_state:
            del st.session_state.preset_question
            
        send_button = st.button("📤 Send", use_container_width=True, type="primary")
        
        # Initialize chat history
        if 'finance_chat_history' not in st.session_state:
            st.session_state.finance_chat_history = []
        
        # Process question
        if send_button and user_question:
            with st.spinner('🤔 Processing...'):
                try:
                    st.session_state.finance_chat_history.append({"role": "user", "content": user_question})
                    
                    # Use the financial agent for chat
                    response = financial_agent.run(user_question)
                    
                    st.session_state.finance_chat_history.append({"role": "ai", "content": response.content})
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Tip: Make sure you have GEMINI_API_KEY set in your .env file")
        
        # Display chat history
        if st.session_state.finance_chat_history:
            st.markdown("---")
            st.subheader("💬 Conversation")
            
            for message in st.session_state.finance_chat_history:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])
            
            # Clear chat button
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.finance_chat_history = []
                st.rerun()

# ============================================================================
# SQL DATABASE CHAT MODE
# ============================================================================
elif app_mode == "🗄️ SQL Database Chat":
    st.title("🗄️ SQL Database Chat")
    st.subheader("Chat with your database using natural language")
    st.markdown("---")
    
    # Database selection in sidebar
    LOCALDB = "USE_LOCALDB"
    MYSQL = "USE_MYSQL"
    
    st.sidebar.subheader("Database Configuration")
    radio_opt = ["Use SQLite Database", "Connect to MySQL"]
    selected_opt = st.sidebar.radio("Database Type", radio_opt)
    
    if radio_opt.index(selected_opt) == 1:
        db_uri = MYSQL
        mysql_host = st.sidebar.text_input("MySQL Host")
        mysql_user = st.sidebar.text_input("MySQL User")
        mysql_password = st.sidebar.text_input("MySQL Password", type="password")
        mysql_db = st.sidebar.text_input("MySQL Database")
    else:
        db_uri = LOCALDB
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.sidebar.error("⚠️ Please set GEMINI_API_KEY in your .env file")
        st.stop()
    
    # Configure database
    @st.cache_resource(ttl="2h")
    def configure_db(uri, host=None, user=None, password=None, dbname=None):
        if uri == LOCALDB:
            dbfile = (Path(__file__).parent / "student.db").absolute()
            
            # Create database if it doesn't exist
            if not dbfile.exists():
                conn = sqlite3.connect(str(dbfile))
                cursor = conn.cursor()
                
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS STUDENT (
                    NAME    VARCHAR(25),
                    CLASS   VARCHAR(25),
                    SECTION VARCHAR(25),
                    MARKS   INT
                )
                """)
                
                records = [
                    ("Aryan", "AI/ML", "A", 100),
                    ("Saransh", "Data Science", "A", 90),
                    ("Bhavya", "French", "B", 50),
                    ("Aditya", "AI/ML", "A", 20),
                    ("Ruchir", "AI/ML", "B", 100),
                    ("Arihant", "French", "B", 100),
                ]
                cursor.executemany("INSERT INTO STUDENT VALUES (?, ?, ?, ?)", records)
                conn.commit()
                conn.close()
            
            creator = lambda: sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True)
            engine = create_engine("sqlite:///", creator=creator)
            return SQLDatabase(engine)
        else:
            if not (host and user and password and dbname):
                st.error("⚠️ Please provide all MySQL connection details")
                st.stop()
            engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{dbname}")
            return SQLDatabase(engine)
    
    # Initialize database
    try:
        if db_uri == MYSQL:
            db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
        else:
            db = configure_db(db_uri)
        
        st.sidebar.success("✅ Database connected!")
        
    except Exception as e:
        st.sidebar.error(f"❌ Connection failed: {str(e)}")
        st.stop()
    
    # LLM and Agent setup - Using Gemini 2.5 Flash
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0,
            convert_system_message_to_human=True
        )
        
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            max_iterations=10,
            max_execution_time=60
        )
        
        executor = AgentExecutor.from_agent_and_tools(
            agent=agent.agent, 
            tools=agent.tools, 
            max_iterations=10
        )
    except Exception as e:
        st.error(f"❌ Failed to initialize SQL agent: {str(e)}")
        st.stop()
    
    # Utility functions
    def get_avg_and_mode(class_name: str):
        try:
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
            
            if avg_df.empty or mode_df.empty:
                return None, None
                
            return float(avg_df["average_marks"].iloc[0]), int(mode_df["mode_marks"].iloc[0])
        except Exception as e:
            st.error(f"Error calculating statistics: {str(e)}")
            return None, None
    
    def format_table_output(response):
        return response
    
    def handle_user_input(query: str):
        q = query.lower()
        avg_mode_match = re.search(r"(average|mode).+class\s+(\w+)", q, re.IGNORECASE)
        
        if avg_mode_match:
            what, cls = avg_mode_match.groups()
            avg, mode = get_avg_and_mode(cls)
            if avg is not None and mode is not None:
                if "average" in what:
                    return f"Class {cls} → Average marks: **{avg:.2f}**"
                if "mode" in what:
                    return f"Class {cls} → Mode marks: **{mode}**"
        
        try:
            cb = StreamlitCallbackHandler(st.container())
            return executor.run(query, callbacks=[cb])
        except Exception as e:
            logging.getLogger(__name__).error("Agent error", exc_info=True)
            return f"Sorry, I couldn't process that query. Error: {str(e)}"
    
    # Chat interface
    st.info("💡 Use natural language to query your database")
    
    # Example questions
    st.write("**Example questions:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👥 Show all students", use_container_width=True):
            st.session_state.sql_preset = "Show all students"
    with col2:
        if st.button("📊 Average marks by class", use_container_width=True):
            st.session_state.sql_preset = "What is the average marks for each class?"
    with col3:
        if st.button("🏆 Top performers", use_container_width=True):
            st.session_state.sql_preset = "Who are the top 3 students by marks?"
    
    st.markdown("---")
    
    # Message history
    if "sql_messages" not in st.session_state or st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.sql_messages = [
            {"role": "assistant", "content": "Hello! I'm your SQL assistant. Ask me anything about the database."}
        ]
    
    # Display chat history
    for msg in st.session_state.sql_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    user_query = st.chat_input("Ask a question about the database...")
    
    # Handle preset questions
    if "sql_preset" in st.session_state:
        user_query = st.session_state.sql_preset
        del st.session_state.sql_preset
    
    # Process user query
    if user_query:
        st.session_state.sql_messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.write(user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = handle_user_input(user_query)
                formatted_response = format_table_output(response)
                st.write(formatted_response)
        
        st.session_state.sql_messages.append(
            {"role": "assistant", "content": formatted_response}
        )

# ============================================================================
# AI ADVISORS MODE
# ============================================================================
else:  # AI Advisors mode
    st.title("🎭 AI Advisors Council")
    st.subheader("Get advice from multiple AI personalities")
    st.markdown("---")
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ Please set GEMINI_API_KEY in your .env file")
        st.stop()
    
    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    
    # Define AI Advisors with their personalities
    AGENTS = {
        "ArthurMorgan": "You are Arthur Morgan, a cowboy from the wild west. You are known for your sharp wit and quick thinking. You are also known for your love of guns and horses. You were in Red Dead Redemption 2. You are considered a loyal friend, helpful guy and a wise man. You help people out of compassion. You have done bad things in the past but you are now trying to make up for your mistakes. The user will be asking you for advice, answer them based on your knowledge and experience. Provide just one short paragraph.",
        
        "Thanos": "You are Thanos, the Mad Titan. You are powerful and determined. You believe that to save the universe from extinction, you need the infinity gauntlet to eliminate half of all life. You don't want to kill anyone, but you are willing to do whatever is necessary to achieve your goal for the greater good. The user will be asking you for advice, answer them based on your philosophy and experience. Provide just one short paragraph.",
        
        "PeterParker": "You are Peter Parker, Spider-Man. You're a young guy from Queens who lives with his aunt. You're very smart and know a lot about science. You got bit by a radioactive spider and gained superpowers. You can swing webs and are very powerful but you always try to do the right thing. You believe that with great power comes great responsibility. You used to be the weak guy, but now you're strong and you help those in need. You believe in second chances. The user will be asking you for advice, answer them based on your values. Provide just one short paragraph.",
        
        "ElonMusk": "You are Elon Musk. You are very smart and ambitious. You are the founder of Tesla and SpaceX, and you purchased X (formerly Twitter). You are always trying to achieve your goals and push the boundaries of technology. You surround yourself with smart people to improve yourself and your ideas. You think about the future of humanity and making life multi-planetary. The user will be asking you for advice, answer them based on your experience in innovation and business. Provide just one short paragraph.",
        
        "DrHouse": "You are Dr. Gregory House. You are brilliant and one of the best diagnosticians in the world. You are driven and always trying to solve complex problems. You don't care about rules and procedures - you do what you think is right. You have friends who back you up when you take unconventional decisions. You are rarely wrong but very stubborn and not afraid to take risks. You use logic and deduction to solve problems. The user will be asking you for advice, answer them with your sharp wit and analytical mind. Provide just one short paragraph.",
        
        "SteveJobs": "You are Steve Jobs. You are a visionary and perfectionist who revolutionized technology and design. You co-founded Apple and created products that changed the world. You believe in simplicity, elegance, and thinking differently. You're demanding but inspire people to do their best work. You focus on the intersection of technology and liberal arts. You think about creating products people don't know they need yet. The user will be asking you for advice, answer them based on your philosophy of innovation and design. Provide just one short paragraph.",
    }


    def get_avatar(agent_name):
        avatars = {
        "ArthurMorgan": "https://i.redd.it/2pm4mgw45qx11.jpg",
        "Thanos": "https://images.steamusercontent.com/ugc/1849286448593117415/1D33B9C8DA3AF6408747794C8678630D64F40415/?imw=512&&ima=fit&impolicy=Letterbox&imcolor=%23000000&letterbox=false",
        "PeterParker": "https://easydrawingguides.com/wp-content/uploads/2024/06/how-to-draw-an-easy-spider-man-featured-image-1200.png",
        "ElonMusk": "https://c.files.bbci.co.uk/7727/production/_103330503_musk3.jpg",
        "DrHouse": "https://media.licdn.com/dms/image/v2/C5612AQGX0elVeRY2Lw/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1520219468256?e=2147483647&v=beta&t=6dXgVRarnn-2hDq4e9WWt6TL_CRfq8s4A0ZguUNPLOg",
        "SteveJobs": "https://platform.theverge.com/wp-content/uploads/sites/2/chorus/uploads/chorus_asset/file/13874040/stevejobs.1419962539.png?quality=90&strip=all&crop=4.3909348441926,0,91.218130311615,100",
    }
        return avatars.get(agent_name, "https://cdn-icons-png.flaticon.com/512/4712/4712109.png")  # Default robot avatar
    
    # Display active advisors in sidebar
    st.sidebar.subheader("🎭 Active Advisors")
    st.sidebar.markdown("---")
    for agent_name in AGENTS.keys():
        avatar = get_avatar(agent_name)
        display_name = agent_name.replace("Dr", "Dr. ")
        st.sidebar.write(f"{display_name}")
    
    # Async function to query individual advisor
    async def ask_advisor(
        system_prompt: str,
        user_prompt: str,
        model: str = "gemini-2.5-flash",
    ) -> str:
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=[system_prompt, user_prompt],
            )
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Query all advisors
    async def query_council(question: str) -> Dict:
        tasks = [ask_advisor(system_prompt, question) for system_prompt in AGENTS.values()]
        answers = await asyncio.gather(*tasks)
        items = [{"agent": name, "answer": ans} for name, ans in zip(AGENTS.keys(), answers)]
        return {"members": items}
    
    # Main interface
    st.info("💡 Ask a question and get advice from all our AI advisors at once!")
    
    # Example questions
    st.write("**Try asking about:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💼 Career advice", use_container_width=True):
            st.session_state.advisor_preset = "I'm feeling stuck in my career. What should I do?"
    with col2:
        if st.button("🎯 Life goals", use_container_width=True):
            st.session_state.advisor_preset = "How do I set and achieve meaningful goals?"
    with col3:
        if st.button("🤔 Decision making", use_container_width=True):
            st.session_state.advisor_preset = "I have a tough decision to make. How should I approach it?"
    
    st.markdown("---")
    
    # Initialize messages
    if "advisor_messages" not in st.session_state:
        st.session_state.advisor_messages = []
    
    # Display message history
    for message in st.session_state.advisor_messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            if message["role"] != "user":
                st.markdown(f"**{message['role']}**")
            st.write(message["content"])
    
    # Chat input
    prompt = st.chat_input("Ask the AI council for advice...")
    
    # Handle preset questions
    if "advisor_preset" in st.session_state:
        prompt = st.session_state.advisor_preset
        del st.session_state.advisor_preset
    
    # Process user input
    if prompt:
        # Add user message
        st.session_state.advisor_messages.append({"role": "user", "content": prompt, "avatar": "👤"})
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)
        
        # Query all advisors
        with st.spinner("🎭 The council is deliberating..."):
            try:
                res = asyncio.run(query_council(prompt))
                
                # Display each advisor's response
                for member in res["members"]:
                    agent_name = member["agent"]
                    answer = member["answer"]
                    avatar = get_avatar(agent_name)
                    
                    st.session_state.advisor_messages.append({
                        "role": agent_name,
                        "content": answer,
                        "avatar": avatar
                    })
                    
                    with st.chat_message(agent_name, avatar=avatar):
                        st.markdown(f"**{agent_name}**")
                        st.write(answer)
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Tip: Make sure you have GEMINI_API_KEY set in your .env file")
    
    # Clear chat button
    if st.session_state.advisor_messages:
        st.markdown("---")
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.advisor_messages = []
            st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.info("🤖 Powered by Gemini 2.5 Flash")