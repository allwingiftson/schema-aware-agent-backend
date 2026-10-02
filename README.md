# Autonomous Schema-Aware Text-to-SQL DB Agent

A production-grade, stateful AI Database Agent backend architecture engineered using **FastAPI** and **LangChain Core Primitives**, powered by high-speed inference execution loops via **Groq Cloud Infrastructure**. 

This system dynamically discovers database topologies, auto-inspects schema structures, generates context-aware dialect-compliant SQL queries, and securely executes data loops against live relational databases on the fly.

## 🏗️ Architectural Topology & Loops
The agent avoids hardcoded variables by employing a multi-turn reasoning workflow:
1. **Dynamic Schema Discovery**: Automatically crawls the target engine metadata layers to register accessible tables.
2. **Structural Inspection**: Interrogates selected tables to map fields, data types, and structural keys.
3. **Execution Safety Loop**: Generates targeted read-only SQL dialects, executing them through bound parameterization interfaces while safely capping token payloads.

## 🛠️ Tech Stack & Engines
* **Web Services**: Python 3.10+, FastAPI, Uvicorn
* **Orchestration**: LangChain Core, LangChain Groq (Model: Qwen/Qwen3.8-27B)
* **Object Relational Engine**: SQLAlchemy v2.0 (PostgreSQL Native Driver integration)
* **Data Layer Validation**: Pydantic v2

## ⚙️ Quick Installation & Setup

1. **Clone & Setup Environment:**
   ```bash
   git clone https://github.com
   cd enterprise-agent-backend
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables (`.env`):**
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   DATABASE_URL=postgresql+psycopg2://username:password@localhost:5432/dbname
   ```

3. **Launch the Engine Service:**
   ```bash
   python main.py
   ```
   Explore the interactive system controller dashboard at: `http://localhost:8000/docs`
