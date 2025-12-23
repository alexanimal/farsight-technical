# Interview Take-Home Project

This take-home project and its functionality has been scoped in a separate document sent to you. Feel free to create any new files and edit existing files as needed - also, feel free to utilize the gpt_chat method in the utils if desired. Make sure to update this README with instructions and important notes before returning the project.

## Project Description

You will be given access to a snippet of our company's proprietary databases that enable you to both semantically search through companies and gather additional metadata on them (more details below). Given these databases, create two workflows exposed via API that each generate an insightful trend based on this data and variable user input. These trends should result in something that would be beneficial for a person working in financial services to know.

**Note:** This project is intended to test your code/system design abilities and is intentionally underspecified to showcase how you make design decisions when faced with uncertainty.

## Requirements

- Provide a brief overview of each trend and why you think it would be interesting to financial professionals
- You can decide the appropriate type of user input required for each trend, but you should be able to demonstrate different results based on different inputs
- Expose each trend via a backend endpoint and return the data necessary for a hypothetical front end to display that trend
- Between the two workflows you implement, you must use a combination of the two types of databases (Pinecone, Postgres)
- Your code must:
  - Include unit tests for different parts of your files, and optionally integration tests.
  - Should be containerized and ready for deployment.
  - For purposes of showcasing technical aptitude, should utilize some sort of distributed task / workflow orchestration framework, like a Celery or Temporal.
- Include a brief writeup (<= 1 paragraph) of how you would deploy this on AWS – what services would you use, how would you spin up different containers, etc. You do not have to implement any CDK code (for your knowledge, Infrastructure + DevOps is not a part of your JD, but being able to prepare code for it is important for streamlining Farsight's development).

## Database Details

Schemas are provided in Excel files – you will need to write your own connectors to plug into these databases:

### PostgreSQL Databases

- **FundingRounds table:** Provides info on company fundraises (e.g. fundraise amounts, valuations, investment dates, etc.)
- **Acquisitions table:** Provides info on company acquisitions (e.g. acquisition date, type of acquisition, acquisition price, etc.)

### Pinecone Vector Database

**Semantic company database:** This is a vector database that clusters companies in the same/similar industries together. It is designed to be queried by the name of a sector and to return a set of companies that closely match that sector. Note: If you are not familiar with Vector DBs, please refer to the first section of this article titled "What is a Vector Database?"

- **Example** (abstracted to show query pattern, refer to provided schema for actual returned information):
  - Query: Closed-source LLM Providers
  - Returns: OpenAI, Anthropic, Google, etc.
- You can additionally pass filters when querying this database that narrow down the query to items which have metadata matching the filters
- This database was created via the text-embedding-3-large model from OpenAI

## Additional Details

- We've provided a barebones starting point for this project containing server.py, utils.py, and .env. Feel free to modify, add, or remove files as you need.
- Inside of the utils.py file, we provided you with a gpt_chat endpoint for interacting with OpenAI's models if you want to include LLM interactions into your workflows. This is completely optional.
  - If you decide to use this, the models you have access to are gpt-4.1-mini and gpt-4.1.
- .env contains the following keys (DO NOT SHARE, these have been generated specifically for your project):
  - `OPENAI_API_KEY`: For use with OpenAI client to interface with their models
  - `PINECONE_API_KEY`: API key for use with the Pinecone client
  - `PINECONE_INDEX`: Name of the pinecone index that stores all the vectors
  - `POSTGRES_DB_NAME`: Name of the postgres database
  - `POSTGRES_USER`: Username for access to the postgres database
  - `POSTGRES_PASSWORD`: Password for access to the postgres database
  - `POSTGRES_HOST`: Host endpoint for connecting to the postgres database
  - `POSTGRES_PORT`: Port number for connecting to the postgres database

## Setup

1. **Create and activate a virtual environment**

   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Start Temporal Server** (Required for workflow orchestration)

   Temporal is a separate service that needs to be running. The easiest way is using Docker Compose:

   ```bash
   # Start Temporal server and its dependencies
   docker-compose up -d

   # Verify Temporal is running
   docker-compose ps

   # View Temporal logs
   docker-compose logs -f temporal
   ```

   This will start:
   - Temporal server on `localhost:7233`
   - Temporal Web UI on `http://localhost:7234`
   - PostgreSQL database for Temporal (on port 5433)

   **Note:** The API will start even if Temporal isn't running, but workflow operations will fail until Temporal is available.

4. **Start the API Server**

   ```bash
   # From the server directory
   uvicorn src.api.api:app --reload
   ```

   The API will be available at `http://localhost:8000`

5. **Start the Temporal Worker** (Required for executing workflows)

   In a separate terminal, start the worker that executes workflows:

   ```bash
   # From the server directory
   python -m src.temporal.worker
   ```

   Or if you have a script entry point:
   ```bash
   python src/temporal/worker.py
   ```

   The worker polls Temporal for workflow and activity tasks and executes them.

## Generating Documentation

This project uses `pdoc` to generate HTML documentation from Python docstrings.

### Prerequisites

**Important:** All project dependencies must be installed before generating documentation, as `pdoc` needs to import the modules to extract full type information and resolve imports.

```bash
# Install dependencies first
pip install -r requirements.txt
```

### Generate Documentation

The easiest way is to use the provided script that automatically discovers all modules:

```bash
# From the server directory
python generate_docs.py
```

This script will:
- Automatically discover all Python modules in `src/` recursively
- Generate HTML documentation for all of them
- Enable search functionality
- Show source code links

**Note:** The script uses file path discovery, so it will document all modules even if some imports fail. However, for best results (full type information, resolved imports), ensure all dependencies are installed.

### Manual Generation

If you prefer to generate docs manually:

```bash
# Use directory path - pdoc will discover all modules recursively
pdoc src --output-dir docs --search --show-source

# Or specify individual modules
pdoc src.config src.api src.agents --output-dir docs --search
```

### Troubleshooting

If you see import errors or warnings:
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Some modules may show warnings if optional dependencies are missing, but documentation will still be generated
- The `src/__init__.py` has an empty `__all__` by design (for Temporal determinism), but the script discovers all submodules automatically

The documentation will be generated in the `docs/` directory. Open `docs/index.html` in your browser to view it.

### View Documentation Locally

After generating, you can view the docs by opening `docs/index.html` in your browser, or serve them with a simple HTTP server:

```bash
# Python 3
python -m http.server 8080 --directory docs

# Then open http://localhost:8080 in your browser
```
