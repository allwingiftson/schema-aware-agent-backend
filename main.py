import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import sqlalchemy

# Load environment variables
load_dotenv()

# Initialize FastAPI App
app = FastAPI(
    title="Schema-Aware DB Agent",
    description="Dynamic database discovery and Text-to-SQL execution loop running on live PostgreSQL.",
    version="2.0.0"
)

# Initialize SQLAlchemy connection engine dynamically from .env
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("Missing DATABASE_URL in environment configuration.")

db_engine = sqlalchemy.create_engine(DATABASE_URL)


# ----------------- INTEL TOOLS FOR DYNAMIC SCHEMA RECOVERY -----------------

@tool
def list_all_tables() -> str:
    """Useful for discovering what table names exist inside the connected database schema."""
    try:
        inspector = sqlalchemy.inspect(db_engine)
        tables = inspector.get_table_names()
        return f"Tables found in database: {', '.join(tables)}" if tables else "No tables found in this database."
    except Exception as e:
        return f"Error fetching tables: {str(e)}"


@tool
def inspect_table_schema(table_name: str) -> str:
    """Useful for inspecting the exact column names, data types, and keys of a specific table."""
    try:
        inspector = sqlalchemy.inspect(db_engine)
        columns = inspector.get_columns(table_name)
        if not columns:
            return f"Table '{table_name}' does not exist or has no columns."

        details = [f"- {col['name']} ({str(col['type'])})" for col in columns]
        return f"Table '{table_name}' structure:\n" + "\n".join(details)
    except Exception as e:
        return f"Error inspecting table '{table_name}': {str(e)}"


@tool
def execute_sql_query(sql_query: str) -> str:
    """Useful for executing raw SELECT SQL queries against the database to fetch real records."""
    try:
        with db_engine.connect() as connection:
            result = connection.execute(sqlalchemy.text(sql_query))
            # Safely capture top 10 rows to avoid blowing up token limits
            rows = result.fetchmany(10)
            if not rows:
                return "Query executed successfully. Zero records returned."

            # Convert row tuples into clean dict strings
            headers = result.keys()
            output = []
            for row in rows:
                row_dict = dict(zip(headers, row))
                output.append(str(row_dict))
            return "\n".join(output)
    except Exception as e:
        return f"Database Execution Error: {str(e)}"


# --------------------------------------------------------------------------

# Set up the Groq Model bound with all 3 dynamic tools
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Missing GROQ_API_KEY in environment configuration.")

llm = ChatGroq(temperature=0, model_name="qwen/qwen3.8-27b")
tools_list = [list_all_tables, inspect_table_schema, execute_sql_query]
llm_with_tools = llm.bind_tools(tools_list)


class QueryRequest(BaseModel):
    prompt: str


class QueryResponse(BaseModel):
    status: str
    agent_response: str


@app.get("/")
def health_check():
    return {"status": "healthy", "database": "connected"}


@app.post("/api/v1/agent/db-chat", response_model=QueryResponse)
async def ask_db_agent(payload: QueryRequest):
    try:
        # Initialize conversation state loop
        messages = [
            HumanMessage(
                content=f"{payload.prompt} (Note: Always explore what tables exist first using tools before running queries).")
        ]

        # Max iteration cycles to prevent infinite logic loops
        for _ in range(5):
            ai_response = llm_with_tools.invoke(messages)

            # If the model does not want to use tools anymore, we have our final answer
            if not ai_response.tool_calls:
                return QueryResponse(status="success", agent_response=str(ai_response.content))

            messages.append(ai_response)

            # Execute requested tool selections sequentially
            for tool_call in ai_response.tool_calls:
                tool_map = {
                    "list_all_tables": list_all_tables,
                    "inspect_table_schema": inspect_table_schema,
                    "execute_sql_query": execute_sql_query
                }

                selected_tool = tool_map.get(tool_call["name"])
                if selected_tool:
                    tool_output = selected_tool.invoke(tool_call["args"])
                    messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))

        # Fallback if iterations cap out
        final_run = llm_with_tools.invoke(messages)
        return QueryResponse(status="success", agent_response=str(final_run.content))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Agent Loop Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
