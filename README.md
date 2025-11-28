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
