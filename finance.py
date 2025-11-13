import streamlit as st
import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.google import Gemini


# Load environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Modern UI with Glassmorphism
css = """
<style>
    /* Modern Glassmorphism Color Palette */
    :root {
        --main-bg: #0f1729;
        --glass-bg: rgba(21, 30, 54, 0.6);
        --glass-card: rgba(25, 34, 60, 0.4);
        --glass-hover: rgba(30, 41, 71, 0.7);
        --accent-primary: #6366f1;
        --accent-secondary: #8b5cf6;
        --accent-tertiary: #ec4899;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --text-tertiary: #94a3b8;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --border: rgba(255, 255, 255, 0.08);
        --shadow: rgba(0, 0, 0, 0.1);
        --glow: rgba(99, 102, 241, 0.5);
    }
    
    /* Base styling with glassmorphism */
    body {
        background: linear-gradient(135deg, var(--main-bg), #131c38) !important;
        color: var(--text-primary);
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
        background-attachment: fixed;
    }
    
    .main {
        background: transparent !important;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }
    
    p, div, li, span {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: var(--glass-card);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        border: 1px solid var(--border);
        box-shadow: 0 8px 32px var(--shadow);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px var(--shadow), 0 0 15px var(--glow);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    /* Gradient accents */
    .gradient-text {
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-tertiary));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-weight: 700;
    }
    
    .gradient-border {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
    }
    
    .gradient-border::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 16px;
        padding: 1px;
        background: linear-gradient(45deg, var(--accent-primary), var(--accent-tertiary));
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        pointer-events: none;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1e5imcs, [data-testid="stSidebar"] {
        background-color: var(--glass-bg) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border-right: 1px solid var(--border) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--glass-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, rgba(99, 102, 241, 0.08), rgba(236, 72, 153, 0.08)) !important;
        border-color: rgba(99, 102, 241, 0.6) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(99, 102, 241, 0.2) !important;
    }
    
    .primary-button > button {
        background: linear-gradient(45deg, var(--accent-primary), var(--accent-secondary)) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: var(--glass-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        padding: 12px 16px !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 2px var(--glow) !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div > div {
        background-color: var(--glass-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
    }
    
    .stSelectbox > div > div > div:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 2px var(--glow) !important;
    }
    
    /* Tables */
    table {
        width: 100%;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        margin: 1em 0 !important;
        border: 1px solid var(--border) !important;
        background: var(--glass-card) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
    }
    
    thead tr th {
        background-color: rgba(30, 41, 71, 0.5) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        text-transform: none !important;
        font-size: 0.875rem !important;
        padding: 14px 18px !important;
        border-bottom: 1px solid var(--border) !important;
    }
    
    tbody tr td {
        background-color: transparent !important;
        color: var(--text-secondary) !important;
        padding: 14px 18px !important;
        border-bottom: 1px solid var(--border) !important;
        font-size: 0.875rem !important;
    }
    
    tbody tr:last-child td {
        border-bottom: none !important;
    }
    
    tbody tr:hover td {
        background-color: rgba(30, 41, 71, 0.3) !important;
    }
    
    /* Finance-specific indicators */
    .positive {
        color: var(--success) !important;
        font-weight: 600 !important;
    }
    
    .negative {
        color: var(--danger) !important;
        font-weight: 600 !important;
    }
    
    .neutral {
        color: var(--warning) !important;
        font-weight: 600 !important;
    }
    
    /* Badge styling */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.3em 0.8em;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 20px;
        background: var(--glass-card);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
    }
    
    .badge-buy {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-sell {
        background: rgba(239, 68, 68, 0.1);
        color: var(--danger);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .badge-hold {
        background: rgba(245, 158, 11, 0.1);
        color: var(--warning);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    /* Gradient divider */
    .gradient-divider {
        height: 1px;
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-tertiary), transparent);
        margin: 1.5rem 0;
        opacity: 0.7;
    }
    
    /* Ticker symbol */
    .ticker {
        font-family: 'JetBrains Mono', 'SF Mono', 'Roboto Mono', monospace;
        font-weight: 700;
        color: var(--accent-primary);
        background: rgba(99, 102, 241, 0.1);
        padding: 4px 8px;
        border-radius: 6px;
    }
    
    /* Price display */
    .price {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-tertiary));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-tertiary)) !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(21, 30, 54, 0.2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.5);
    }
    
    /* Stat cards with glassmorphism */
    .stat-group {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-top: 16px;
    }
    
    .stat-card {
        background: var(--glass-card);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid var(--border);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12), 0 0 10px var(--glow);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
        font-weight: 600;
    }
    
    .stat-value {
        font-size: 1.75rem;
        font-weight: 700;
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-tertiary));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        line-height: 1.2;
    }
    
    /* Chat interface */
    .chat-container {
        margin-top: 24px;
        display: flex;
        flex-direction: column;
    }
    
    .chat-message {
        display: flex;
        margin-bottom: 20px;
    }
    
    .chat-message-ai {
        flex-direction: row;
    }
    
    .chat-message-user {
        flex-direction: row-reverse;
    }
    
    .chat-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        flex-shrink: 0;
    }
    
    .chat-avatar-ai {
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        color: white;
        margin-right: 14px;
    }
    
    .chat-avatar-user {
        background: linear-gradient(135deg, var(--accent-secondary), var(--accent-tertiary));
        color: white;
        margin-left: 14px;
    }
    
    .chat-bubble {
        padding: 16px 20px;
        border-radius: 12px;
        max-width: 80%;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    
    .chat-bubble-ai {
        background: var(--glass-card);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid var(--border);
        color: var(--text-primary);
    }
    
    .chat-bubble-user {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2));
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        color: var(--text-primary);
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    /* Spinner animation */
    .loading-spinner {
        display: inline-block;
        width: 24px;
        height: 24px;
        border: 2px solid rgba(99, 102, 241, 0.1);
        border-radius: 50%;
        border-top-color: var(--accent-primary);
        animation: spin 0.8s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Navigation pills */
    .nav-pills {
        display: flex;
        gap: 8px;
        margin-bottom: 24px;
        padding: 4px;
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid var(--border);
    }
    
    .nav-pill {
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .nav-pill-active {
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
        color: white;
        box-shadow: 0 4px 8px rgba(99, 102, 241, 0.25);
    }
    
    /* Logo and brand */
    .logo {
        font-weight: 700;
        font-size: 1.3rem;
        display: flex;
        align-items: center;
    }
    
    .logo-icon {
        margin-right: 10px;
        font-size: 1.4rem;
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-tertiary));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    
    /* Quick action buttons */
    .quick-actions {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin: 16px 0;
    }
    
    .quick-action {
        background: var(--glass-card);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px 18px;
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary);
        transition: all 0.2s ease;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .quick-action:hover {
        background: linear-gradient(45deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.08));
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1), 0 0 5px var(--glow);
    }
    
    .quick-action-icon {
        color: var(--accent-primary);
    }
    
    /* Feedback tags */
    .feedback-tag {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        background: var(--glass-card);
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid var(--border);
    }
    
    .feedback-tag:hover {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.08));
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    /* Search bar */
    .search-container {
        position: relative;
        margin: 16px 0;
    }
    
    .search-input {
        width: 100%;
        padding: 14px 20px;
        padding-left: 45px;
        border-radius: 12px;
        background: var(--glass-card);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid var(--border);
        color: var(--text-primary);
        font-size: 1rem;
        transition: all 0.2s ease;
    }
    
    .search-input:focus {
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 0 0 3px var(--glow);
        outline: none;
    }
    
    .search-icon {
        position: absolute;
        left: 15px;
        top: 50%;
        transform: translateY(-50%);
        color: var(--text-tertiary);
    }
    
    /* Help tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip-text {
        visibility: hidden;
        position: absolute;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid var(--border);
        color: var(--text-primary);
        padding: 10px 14px;
        border-radius: 8px;
        width: 200px;
        font-size: 0.75rem;
        z-index: 1;
        opacity: 0;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
</style>
"""

# Streamlit Page Config
st.set_page_config(
    page_title="Stock Market Screener",
    page_icon="✨",
    layout="wide"
)

# Inject custom CSS
st.markdown(css, unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
    <div class="logo">
        <span class="logo-icon">✨</span>
        <span class="gradient-text">Quantum Finance</span>
    </div>
    <div style="font-size: 0.85rem; color: var(--text-tertiary); display: flex; align-items: center; gap: 8px;">
        <span style="background: rgba(99, 102, 241, 0.1); padding: 4px 8px; border-radius: 6px; display: flex; align-items: center; gap: 6px;">
            <span style="width: 8px; height: 8px; background: #10b981; border-radius: 50%; display: inline-block;"></span>
            Live Data
        </span>
        <span>Powered by AI</span>
    </div>
</div>
<div class="gradient-divider"></div>
""", unsafe_allow_html=True)

# --- Define AI Agents ---

## Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the latest information",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[DuckDuckGoTools()],
    instructions=[
        "ALWAYS present information in tabular format where possible",
        "Always include sources with dates of publication",
        "Structure your output with clear headings and bullet points",
        "For financial news, categorize information by market impact (Positive/Neutral/Negative) in a table format",
        "Include a summary of key takeaways at the end in a table format"
    ],
    show_tool_calls=True,
    markdown=True,
)

## Financial Agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True),
    ],
    instructions=[
        "ALWAYS present ALL data in tabular format - no exceptions",
        "Present analyst recommendations with consensus ratings in a table (Strong Buy/Buy/Hold/Sell/Strong Sell)",
        "Include target price ranges and average price targets in a dedicated table",
        "Provide technical indicators with clear buy/sell signals in a table format",
        "Format all price data with appropriate currency symbols",
        "Present 'Timing Guidance' section with short-term, medium-term, and long-term outlooks in a table",
        "Add a 'Risk Assessment' section in tabular format highlighting potential downsides",
        "Structure output with clear headings: Summary, Price Data, Fundamentals, Analyst Views, Technical Analysis, Timing Guidance, Risk Assessment",
        "Even summary information must be presented in a table format"
    ],
    show_tool_calls=True,
    markdown=True,
)

## Chatbot Agent with Web Search Capability
# chatbot_agent = Agent(
#     name="Finance Chatbot Agent",
#     model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
#     tools=[DuckDuckGoTools()],
#     instructions=[
#         "ALWAYS organize and present data in tables wherever possible",
#         "Provide clear and concise answers to finance-related questions",
#         "Include sources when fetching real-time data",
#         "Structure responses with headings and tables - avoid bullet points where tables can be used instead",
#         "For investment advice, always include timing considerations and risk factors in tabular format",
#         "Use emojis sparingly to highlight important points",
#         "Include a 'Key Takeaways' section at the end of comprehensive responses in a table format"
#     ],
#     show_tool_calls=True,
#     markdown=True,
# )

# multi_chatbot_agent = Agent(
#     team=[chatbot_agent, web_search_agent],
#     model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
#     instructions=[
#         "ALWAYS present information in tables - consider this a strict requirement",
#         "Always include sources with dates in tabular format",
#         "Structure output with clear headings and tables for each section",
#         "First use Finance Chatbot Agent to answer the question",
#         "Then use the Web Search Agent for recent news",
#         "Present a unified analysis that combines fundamental data with recent news in tabular format",
#         "Include timing guidance (when to buy/sell) based on technical indicators and news sentiment in a table",
#         "Clearly separate fact-based information from AI-generated analysis using separate tables",
#         "End with actionable insights and key takeaways presented in a table"
#     ],
#     show_tool_calls=True,
#     markdown=True,
# )


# Combined Multi-Agent System
multi_ai_agent = Agent(
    team=[finance_agent, web_search_agent],
    model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
    instructions=[
        "ALWAYS present ALL information in table format - this is mandatory",
        "Structure output with clear sections using markdown headings, with each section containing at least one table",
        "First use the Finance Agent to get detailed stock data",
        "Then use the Web Search Agent for recent news and market sentiment",
        "Present ALL data in tables - never use paragraphs where tables can be used instead",
        "Include a 'Stock Fundamentals' table with key metrics and comparisons to industry averages",
        "Provide 'Analyst Consensus' table with specific ratings, target prices and timeframes",
        "Add 'Technical Analysis' table with key indicators and clear buy/sell signals",
        "Include 'Entry Points' table suggesting optimal buying opportunities based on technical patterns",
        "Add 'Investment Timeframe' table (Short-term trader vs. Long-term investor recommendations)",
        "Include 'Risk Assessment' table highlighting potential downside scenarios",
        "End with 'Action Plan' table summarizing recommendations with clear timing guidance",
        "Always cite sources for all external information in a dedicated sources table"
    ],
    show_tool_calls=True,
    markdown=True,
)

# --- Streamlit UI ---
# Create a container for the main content
main_container = st.container()

# Simplified tabbed interface with fewer options
tabs = st.tabs(["📊 Market Analysis", "💬 AI Assistant"])

with tabs[0]:  # Market Analysis Tab
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Simplified header
        st.markdown("""
        <div class="glass-card">
            <h2>AI Stock Analysis</h2>
            <p>Get AI-powered insights on any publicly traded company.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Stock lookup - simplified
        stock_symbol = st.text_input("Enter Ticker Symbol", value="AAPL", placeholder="E.g., AAPL, MSFT")
        
        # Simplified analysis options
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Complete Analysis", "News Impact"]
        )
        
        analyze_button = st.button("Analyze", use_container_width=True)
    
    # Popular stocks as quick buttons - simplified
    st.markdown("<p style='margin-top: 16px; font-size: 0.9rem;'>Popular stocks:</p>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.button("AAPL", use_container_width=True)
    with col2:
        st.button("MSFT", use_container_width=True)
    with col3:
        st.button("GOOGL", use_container_width=True)
    with col4:
        st.button("TSLA", use_container_width=True)
    
    # Display analysis results
    if analyze_button or stock_symbol:
        with st.spinner('Analyzing market data...'):
            try:
                # Determine analysis type and create prompt
                if analysis_type == "Complete Analysis":
                    prompt = f"Provide comprehensive analysis for {stock_symbol} including current price, analyst recommendations, technical indicators, and investment outlook."
                else:
                    prompt = f"Find and summarize the latest news for {stock_symbol} with market impact assessment."
                
                # Result header
                st.markdown(f"""
                <div class="glass-card">
                    <h3><span class="ticker">${stock_symbol}</span> Analysis</h3>
                    <div class="gradient-divider"></div>
                """, unsafe_allow_html=True)
                
                # Generate analysis
                response = multi_ai_agent.run(prompt)
                st.markdown(response.content)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error: {e}")

with tabs[1]:  # AI Assistant Tab
    st.markdown("""
    <div class="glass-card">
        <h2>AI Financial Assistant</h2>
        <p>Ask questions about markets, stocks, or investment strategies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple example questions
    st.markdown("<p style='margin-top: 16px; font-size: 0.9rem;'>Try asking about:</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("Investment tips", use_container_width=True)
    with col2:
        st.button("Crypto market outlook", use_container_width=True)
    
    # Input and send button - simplified
    user_question = st.text_input("Ask anything about finance", placeholder="e.g., How do interest rates affect stocks?")
    send_button = st.button("Send", use_container_width=True)
    
    # Initialize chat history in session state if not already there
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Process user question
    if send_button and user_question:
        with st.spinner('Processing...'):
            try:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                # Get AI response
                gemini_chat_model = Gemini(id="gemini-2.0-flash")
                ai_response = gemini_chat_model.chat(user_question)
                
                # Add AI response to history
                st.session_state.chat_history.append({"role": "ai", "content": ai_response.content})
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Display chat history - simplified
    if st.session_state.chat_history:
        st.markdown('<div class="glass-card" style="padding: 20px;">', unsafe_allow_html=True)
        
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div style="background: rgba(99, 102, 241, 0.1); padding: 10px 16px; border-radius: 8px; margin-bottom: 12px; border: 1px solid rgba(99, 102, 241, 0.2); text-align: right;">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: var(--glass-card); padding: 10px 16px; border-radius: 8px; margin-bottom: 16px; border: 1px solid var(--border);">
                    <strong>AI:</strong>
                """, unsafe_allow_html=True)
                
                st.markdown(message["content"])
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Simple footer
st.markdown("""
<div style="margin-top: 2rem;">
    <div class="gradient-divider"></div>
    <div style="display: flex; justify-content: center; margin-top: 1rem;">
        <div style="font-size: 0.8rem; color: var(--text-tertiary);">
            Powered by AI • Data refreshed in real-time
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize current time
if 'current_time' not in st.session_state:
    from datetime import datetime
    st.session_state.current_time = datetime.now().strftime("%H:%M:%S")

# Add spinner style for animations
st.markdown("""
<style>
@keyframes spinner {
    to {transform: rotate(360deg);}
}
</style>
""", unsafe_allow_html=True)