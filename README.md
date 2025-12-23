# Farsight Technical

A full-stack application for financial data analysis and trend generation, featuring a FastAPI backend with Temporal workflow orchestration and a React frontend.

## Project Structure

```text
farsight-technical/
├── server/          # Python FastAPI backend
│   ├── src/         # Source code
│   ├── tests/       # Test suite
│   └── pyproject.toml
├── ui/              # React/Vite frontend
│   ├── app/         # React application
│   └── package.json
└── docker-compose.yml  # Docker orchestration
```

## Prerequisites

### Required Software

- **Python 3.10+** (see `server/.python-version`)
- **Node.js 16+** and npm
- **Docker** and **Docker Compose** (for containerized deployment)
- **PostgreSQL** (for main database - provided externally or via Docker)
- **Redis** (for conversation history - included in docker-compose)

### Required API Keys

You'll need the following API keys (provided in `.env` file):

- `OPENAI_API_KEY` - OpenAI API key for LLM operations
- `PINECONE_API_KEY` - Pinecone API key for vector database
- `PINECONE_INDEX` - Pinecone index name
- PostgreSQL connection details (host, port, database, user, password)

## Quick Start with Docker

The easiest way to run the entire application:

1. **Create environment file**

   Create `server/.env` with your API keys and database credentials:

   ```bash
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Pinecone Configuration
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_INDEX=your_pinecone_index_name
   
   # PostgreSQL Configuration
   POSTGRES_HOST=your_postgres_host
   POSTGRES_PORT=5432
   POSTGRES_DB_NAME=your_database_name
   POSTGRES_USER=your_postgres_user
   POSTGRES_PASSWORD=your_postgres_password
   
   # API Security
   API_KEY=1234567890  # Change this to a secure key
   
   # Optional: Langfuse Configuration
   LANGFUSE_SECRET_KEY=
   LANGFUSE_PUBLIC_KEY=
   LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
   
   # Redis Configuration (for Docker, defaults are fine)
   REDIS_HOST=redis
   REDIS_PORT=6379
   ```

2. **Start all services**

   ```bash
   docker-compose up -d
   ```

   This starts:
   - Temporal server (workflow orchestration) on `localhost:7233`
   - Temporal UI on `http://localhost:7234`
   - Redis (conversation history) on `localhost:6379`
   - API server on `http://localhost:8000`
   - Temporal worker (executes workflows)
   - UI frontend on `http://localhost:3000`

3. **Verify services**

   ```bash
   # Check all containers are running
   docker-compose ps
   
   # Check API health
   curl http://localhost:8000/health
   
   # Check UI health
   curl http://localhost:3000/health
   ```

4. **Access the application**

   - Frontend UI: <http://localhost:3000>
   - API Documentation: <http://localhost:8000/docs>
   - Temporal UI: <http://localhost:7234>

## Manual Setup (Development)

### Server Setup

1. **Navigate to server directory**

   ```bash
   cd server
   ```

2. **Create and activate virtual environment**

   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate virtual environment
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**

   The project uses `uv` for dependency management. Install dependencies:

   ```bash
   # Using uv (recommended)
   uv pip install -e .

   # Or using pip
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Create `.env` file**

   Create `server/.env` with your configuration (see Quick Start section above).

5. **Start Temporal Server** (Required for workflow orchestration)

   ```bash
   # From project root
   docker-compose up -d temporal temporal-ui temporal-postgresql redis
   
   # Or start all services
   docker-compose up -d
   ```

   This starts:
   - Temporal server on `localhost:7233`
   - Temporal Web UI on `http://localhost:7234`
   - PostgreSQL for Temporal (on port 5433)
   - Redis on `localhost:6379`

6. **Start the API Server**

   ```bash
   # From the server directory
   uvicorn src.api.api:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at `http://localhost:8000`

   - API docs: <http://localhost:8000/docs>
   - Health check: <http://localhost:8000/health>

7. **Start the Temporal Worker** (Required for executing workflows)

   In a separate terminal:

   ```bash
   # From the server directory
   python -m src.temporal.worker
   ```

   The worker polls Temporal for workflow and activity tasks and executes them.

### UI Setup

1. **Navigate to UI directory**

   ```bash
   cd ui
   ```

2. **Install dependencies**

   ```bash
   npm install
   ```

3. **Configure environment variables** (optional)

   Create `ui/.env` or `ui/.env.local`:

   ```bash
   VITE_API_BASE_URL=http://localhost:8000
   VITE_API_KEY=1234567890
   ```

   These are optional - defaults are set in `ui/app/api/chatAPI.ts`.

4. **Start development server**

   ```bash
   npm run dev
   ```

   The UI will be available at `http://localhost:5173` (Vite default port)

5. **Build for production**

   ```bash
   npm run build
   ```

   Built files will be in `ui/dist/`

## Development Workflow

### Running Tests

**Server tests:**

```bash
cd server

# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific test file
pytest tests/path/to/test_file.py

# Run with verbose output
pytest -v
```

**UI tests:**

```bash
cd ui

# Run linter
npm run lint
```

### Code Formatting

**Server:**

```bash
cd server

# Format code with black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Type checking with mypy
mypy src/
```

### Development Mode

1. **Start infrastructure services** (Temporal, Redis):

   ```bash
   docker-compose up -d temporal temporal-ui temporal-postgresql redis
   ```

2. **Start API server** (with auto-reload):

   ```bash
   cd server
   uvicorn src.api.api:app --reload
   ```

3. **Start Temporal worker** (in separate terminal):

   ```bash
   cd server
   python -m src.temporal.worker
   ```

4. **Start UI** (with hot reload):

   ```bash
   cd ui
   npm run dev
   ```

## Configuration

### Server Configuration

Server configuration is managed via environment variables loaded from `server/.env`. Key settings:

- **Temporal**: `TEMPORAL_ADDRESS`, `TEMPORAL_NAMESPACE`, `TEMPORAL_TASK_QUEUE`
- **PostgreSQL**: `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB_NAME`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
- **Pinecone**: `PINECONE_API_KEY`, `PINECONE_INDEX`
- **OpenAI**: `OPENAI_API_KEY`
- **Redis**: `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD`
- **API Security**: `API_KEY` (used for API authentication)

See `server/src/config.py` for all available configuration options.

### UI Configuration

UI configuration is managed via Vite environment variables (prefixed with `VITE_`):

- `VITE_API_BASE_URL` - Backend API URL (default: `http://localhost:8000`)
- `VITE_API_KEY` - API key for authentication (default: `1234567890`)

These can be set in:

- `ui/.env` or `ui/.env.local` files
- Docker build arguments (see `docker-compose.yml`)

## Docker Services

The `docker-compose.yml` file defines the following services:

- **temporal**: Temporal workflow server
- **temporal-ui**: Temporal web UI
- **temporal-postgresql**: PostgreSQL database for Temporal
- **redis**: Redis for conversation history
- **server**: FastAPI backend server
- **worker**: Temporal worker for executing workflows
- **ui**: React frontend served via nginx

### Docker Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f [service-name]

# Stop all services
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# Stop and remove volumes
docker-compose down -v
```

## Architecture

### Backend

- **FastAPI**: REST API framework
- **Temporal**: Workflow orchestration for distributed task execution
- **PostgreSQL**: Main database for funding rounds and acquisitions data
- **Pinecone**: Vector database for semantic company search
- **Redis**: Conversation history and caching
- **OpenAI**: LLM operations for agent responses

### Frontend

- **React 19**: UI framework
- **Vite**: Build tool and dev server
- **TypeScript**: Type safety
- **TailwindCSS**: Styling
- **Redux Toolkit**: State management
- **React Query**: Data fetching and caching
- **React Router**: Client-side routing

### Workflow System

The application uses Temporal for workflow orchestration:

1. API receives task request
2. Creates Temporal workflow
3. Worker executes workflow activities
4. Agents process tasks using tools (database queries, LLM calls, etc.)
5. Results streamed back via Server-Sent Events (SSE)

## Troubleshooting

### API won't start

- Check that Temporal is running: `docker-compose ps temporal`
- Verify `.env` file exists and has required keys
- Check logs: `docker-compose logs server`

### Worker not processing tasks

- Ensure worker is running: `docker-compose ps worker`
- Check Temporal connection: `docker-compose logs worker`
- Verify Temporal server is accessible: `curl http://localhost:7233`

### UI can't connect to API

- Verify API is running: `curl http://localhost:8000/health`
- Check `VITE_API_BASE_URL` in UI environment
- Check CORS settings in API middleware

### Database connection errors

- Verify PostgreSQL is accessible
- Check connection credentials in `.env`
- Ensure database exists and is accessible from server

## Additional Resources

- **Server README**: See `server/README.md` for detailed server documentation
- **UI README**: See `ui/README.md` for UI-specific details
- **Temporal Documentation**: <https://docs.temporal.io/>
- **FastAPI Documentation**: <https://fastapi.tiangolo.com/>
- **Vite Documentation**: <https://vitejs.dev/>
