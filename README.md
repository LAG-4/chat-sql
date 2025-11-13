# AI Assistant Hub

A powerful multi-purpose AI application built with Streamlit that combines financial analysis, database querying, and AI advisory capabilities - all powered by Google's Gemini 2.5 Flash model.

## Overview

AI Assistant Hub is an all-in-one intelligent platform featuring three distinct modes:

1. **Financial Analysis** - Real-time stock market analysis with AI-powered insights
2. **SQL Database Chat** - Natural language interface for database queries
3. **AI Advisors Council** - Get diverse perspectives from multiple AI personalities

## Features

### =� Financial Analysis Mode

- **Real-time Stock Data**: Get current prices, fundamentals, and key metrics for any stock
- **Comprehensive Analysis**: P/E ratios, market cap, revenue, and technical indicators
- **News & Sentiment**: Web search integration for latest market news and sentiment analysis
- **Analyst Recommendations**: View consensus ratings and target prices
- **AI Assistant**: Chat-based interface for investment questions and strategies
- **Quick Access**: Pre-configured buttons for popular stocks (AAPL, MSFT, GOOGL, TSLA)

**Analysis Types:**
- Complete Analysis (full fundamental + technical + news)
- News Impact (latest news with sentiment analysis)
- Quick Overview (essential metrics at a glance)

### =� SQL Database Chat Mode

- **Natural Language Queries**: Ask questions in plain English, get SQL results
- **Database Support**: SQLite (local) and MySQL (remote) connections
- **Intelligent Agent**: LangChain-powered SQL agent with error handling
- **Sample Database**: Pre-configured student database for testing
- **Query Examples**: Built-in example questions for quick start
- **Visual Feedback**: Streamlit callback handlers for query execution visibility

**Supported Queries:**
- Data retrieval (e.g., "Show all students")
- Aggregations (e.g., "What is the average marks for each class?")
- Filtering (e.g., "Who are the top 3 students by marks?")
- Statistical analysis (average, mode, count, etc.)

### <� AI Advisors Council Mode

Get advice from six distinct AI personalities, each with unique perspectives:

- **Arthur Morgan** - Wild West cowboy with wisdom and compassion
- **Thanos** - The Mad Titan's philosophy on difficult decisions
- **Peter Parker (Spider-Man)** - Young hero balancing responsibility and power
- **Elon Musk** - Innovation and entrepreneurship from a tech visionary
- **Dr. Gregory House** - Analytical brilliance and unconventional problem-solving
- **Steve Jobs** - Design thinking and revolutionary product vision

**Features:**
- Parallel querying of all advisors
- Unique avatars for each personality
- Persistent conversation history
- Example questions for common scenarios

## Installation

### Prerequisites

- Python 3.12 or higher
- Google Gemini API key

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd chat-sql
   ```

2. **Install dependencies**

   Using `uv` (recommended):
   ```bash
   uv pip install -e .
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

   To get a Gemini API key:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy and paste it into your `.env` file

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the application**

   Open your browser and navigate to `http://localhost:8501`

## Configuration

### Database Configuration (SQL Mode)

**SQLite (Default)**
- Automatically creates a sample `student.db` database
- No additional configuration required
- Perfect for testing and development

**MySQL**
- Select "Connect to MySQL" in the sidebar
- Provide connection details:
  - Host (e.g., localhost)
  - Username
  - Password
  - Database name

### API Keys

The application requires a Google Gemini API key. Set it in your `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

## Usage

### Financial Analysis

1. Select **=� Financial Analysis** from the sidebar
2. Choose between:
   - **=� Market Analysis**: Analyze specific stocks
     - Enter a ticker symbol (e.g., AAPL)
     - Select analysis type
     - Click "Analyze"
   - **=� AI Assistant**: Ask general financial questions
     - Type your question
     - Get AI-powered insights

### SQL Database Chat

1. Select **=� SQL Database Chat** from the sidebar
2. Configure your database (SQLite or MySQL)
3. Ask questions in natural language:
   - "Show all students"
   - "What is the average marks for AI/ML class?"
   - "Who scored above 90?"
4. View results in structured format

### AI Advisors Council

1. Select **<� AI Advisors** from the sidebar
2. Ask a question about:
   - Career decisions
   - Life goals
   - Problem-solving
   - Innovation and creativity
3. Get responses from all six advisors simultaneously
4. Compare different perspectives and approaches

## Project Structure

```
chat-sql/


