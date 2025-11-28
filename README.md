# Malaria LLM Assistant
A lightweight LLM-powered tool for querying, summarizing, and interpreting integrated malaria genomic prevalence data. The assistant converts natural-language questions into structured queries, extracts relevant data from pf7, WHO, WWARN, and systematic-review datasets, and generates clear scientific narratives.

# Installation
~~~
git clone git@github.com:cecile-meier-scherling/malaria-genomics-LLM.git
cd malaria-genomics-LLM
~~~
To install the dependencies:
~~~
conda env create -f environment.yml
conda activate malaria-genomics-llm
~~~
To set up your OpenAI key:
```
export OPENAI_API_KEY="your_key_here"
```

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
