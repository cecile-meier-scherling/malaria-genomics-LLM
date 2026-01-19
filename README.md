# Malaria LLM Assistant
A lightweight command-line tool for asking natural-language questions about **pre-computed malaria resistance prevalence data** (CSV/Parquet).  
The tool:
1) converts a question into a structured filter (country/mutation/year range),
2) filters your dataset,
3) summarizes trends by year/site/study,
4) asks a **local LLM via Ollama** to produce a concise scientific narrative.

## Features
- Works with `.csv` or `.parquet`
- Structured query parsing (country, mutation, year_min, year_max)
- Weighted prevalence summaries (by sample size)
- Local inference with **Ollama** (no cloud API required)

---

## Quickstart (recommended)

### 1) Install Ollama and pull a model
Install Ollama: https://ollama.com

Then pull a model (default: `llama3`):
```bash
ollama pull llama3
```

### 2) Install this package
~~~
git clone git@github.com:cecile-meier-scherling/malaria-genomics-LLM.git
cd malaria-genomics-LLM
pip install -e .
~~~
### 3) Install the dependencies:
~~~
conda env create -f environment.yml
conda activate malaria-genomics-llm
~~~

### 4) Run the interactive prompt with data
The code has two arguments: data and question. If the data is not provided as an argument, the model uses the data path defined in the config file.
~~~
malaria-genomics-llm --data data/raw/all_who_get_prevalence.csv --question "How has 675V prevalence changed over time in Uganda?"
~~~

Data requirements: Your input file must contain (after loading/renaming) these columns:
- country (string)
- site (string)
- year (int) : sampling/collection year
- mutation (string) : gene:position:aa (e.g. k13:675:V)
- prevalence (float)
- n_samples (int)
- study_id (string or int)
- authors (string)
- year_pub (int)
- url (string)

### 5) Example Questions
Example questions:
- “How has 675V prevalence changed over time in Uganda?”
- “Compare 675V prevalence across sites in Rwanda from 2015 to 2020.”
- “What studies report 675V in Ethiopia?”

---
# Project Architecture
Malaria genomic surveillance data is rich but difficult to navigate across multiple sources.
This project provides a simple interface to:
1. Ask questions in natural language: _What is the trend of K13 561H in Uganda after 2012?_
2. Translate the questions into structured filters using an LLM
   ```
   {country: "Uganda", mutation: "K13_561H", year_min: 2012}
   ```
3. Extract and summarize relevant data from the integrated database, such as prevalence over time, study-level contributions, geographic differences, and sample sizes and caveats.
4. Generate a scientific interpretation using an LLM: Clear narrative using only the provided numeric summaries and study metadata.

```
question (NL)
     ↓
LLM → structured query (JSON)
     ↓
data filter (deterministic)
     ↓
data summary (numeric + study metadata)
     ↓
LLM → scientific narrative summary
```

# License
MIT License
