# 🧙 DataWizard — AI-Powered Data Analysis & SQL Agent

A conversational data science assistant built with Streamlit and LangGraph. Ask questions in plain English — the agent decides which tools to call, runs them, and renders results as interactive tables, charts, and code snippets.

**Live demo:** [Deployed on Streamlit Community Cloud]

---

## Features

| Capability | What you can ask |
|---|---|
| **SQL queries** | "Show total revenue by region", "How many customers are on each plan?" |
| **CSV analysis** | "Analyze sample_regression.csv", "What's the distribution of salary?" |
| **ML training** | "Train a classification model predicting defaulted with 200 trees" |
| **Visualizations** | "Plot a correlation heatmap", "Show a scatter of income vs credit score" |
| **Step-by-step reasoning** | Every response has a 🧠 toggle showing the agent's tool calls |
| **Equivalent Python code** | Every result has a 🐍 toggle with copy-pasteable pandas/sklearn/SQL code |

---

## Project Structure

```
├── app.py                      # Streamlit UI and chat loop
├── agent.py                    # LangGraph agent setup (create_agent + MemorySaver)
├── store.py                    # Thread-safe module-level store (files, plots, tables, code)
├── tools/
│   ├── csv_tools.py            # list_datasets, load_and_cache_dataset, analyze_dataset,
│   │                           # train_model, plot_dataset
│   └── sql_tools.py            # get_sql_schema, run_sql_query
├── data/
│   ├── sample.db               # SQLite DB: sales, customers, monthly_targets
│   ├── sample_regression.csv   # Salary prediction dataset (300 rows)
│   └── sample_classification.csv  # Loan default dataset (300 rows)
├── seed_db.py                  # Creates and populates sample.db
├── generate_sample_csvs.py     # Generates sample CSV files
├── requirements.txt
├── runtime.txt                 # Pins Python 3.11 for Streamlit Cloud
└── .streamlit/
    └── secrets.toml            # GROQ_API_KEY (gitignored)
```

---

## Quickstart (local)

**1. Clone and set up the environment**
```bash
git clone https://github.com/YOUR_USERNAME/datawizard.git
cd datawizard

python3 -m venv datawizard-env
source datawizard-env/bin/activate   # Windows: datawizard-env\Scripts\activate
pip install -r requirements.txt
```

**2. Add your Groq API key**

Get a free key at [console.groq.com](https://console.groq.com), then:
```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "your-key-here"
```

**3. Generate sample data**
```bash
python generate_sample_csvs.py
python seed_db.py
```

**4. Run**
```bash
streamlit run app.py
```

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Set main file to `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your-key-here"
   ```
5. Deploy — `runtime.txt` pins Python 3.11 so all wheels install cleanly

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Agent framework | LangGraph (`create_agent` + `MemorySaver`) |
| LLM | Groq — `meta-llama/llama-4-scout-17b-16e-instruct` |
| Data | pandas, numpy |
| ML | scikit-learn (RandomForest) |
| Visualizations | matplotlib, seaborn |
| Database | SQLite (stdlib) |

---

## Sample Questions to Try

**SQL**
- "What tables are in the database?"
- "Show total revenue by region"
- "Which product has the highest average quantity sold?"
- "Compare revenue vs targets for each region"

**CSV Analysis**
- "Analyze sample_classification.csv"
- "What are the top correlations in the diabetes dataset?"

**ML Training**
- "Train a classification model on sample_classification.csv predicting defaulted"
- "Retrain with 300 trees, 25% test split, and max depth 10"
- "Train a regression model to predict salary"

**Visualizations**
- "Plot a correlation heatmap for sample_regression.csv"
- "Show a histogram of BMI"
- "Scatter plot of glucose vs insulin"

---

## Architecture Notes

- **Thread-safe store**: LangGraph runs tool calls in a `ThreadPoolExecutor`. `store.py` uses `queue.Queue` to pass plots, DataFrames, and code snippets from worker threads back to the Streamlit main thread.
- **No nested LLM calls**: `run_sql_query` accepts a SQL string the agent writes itself — avoids the double-LLM pattern that caused Groq tool-call failures.
- **Chain-of-Thought prompting**: System prompt enforces a plan-then-execute reasoning protocol with explicit tool-ordering rules.
