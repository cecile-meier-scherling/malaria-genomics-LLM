import pandas as pd
from typing import Dict, Any
import json

from src import config

def llm_generate_answer(
    question: str,
    query: Dict[str, Any],
    summary: Dict[str, Any],
) -> str:
    """
    Ask the local LLM to produce a scientific-style narrative based ONLY
    on the provided summary.
    """
    system_msg = (
        "You are an assistant helping a malaria genomicist interpret pre-computed "
        "prevalence data and study metadata. You must ONLY use the numeric summaries "
        "and study information provided. Do not invent new data or new studies. "
        "If data are sparse or missing, say so explicitly. "
        "Write clearly in a scientific but concise style."
    )

    context = {
        "original_question": question,
        "parsed_query": query,
        "data_summary": summary,
    }

    user_msg = (
        "You are given structured summaries of malaria resistance prevalence data "
        "for a specific user question.\n\n"
        "The JSON below has the following structure:\n"
        "- original_question: the user's natural language question.\n"
        "- parsed_query: the structured filter (country, mutation, year range, etc.).\n"
        "- data_summary:\n"
        "    - has_data: whether any data exist for this query.\n"
        "    - yearly: list of records {year, n_samples, mean_prevalence}. "
        "      This is the primary source for temporal trends.\n"
        "    - by_site: list of records {site, country, n_samples, mean_prevalence, "
        "      first_year, last_year}. This is the primary source for spatial "
        "      and geographic variation.\n"
        "    - by_study: list of study-level records "
        "      {study_id, authors, year_pub, n_samples, year_min, year_max, mean_prev}. "
        "      These provide supporting evidence about consistency across studies.\n\n"
        "Here is the structured data:\n\n"
        + json.dumps(context, indent=2)
        + "\n\n"
        "Before writing the summary, extract the mutation name from parsed_query['mutation'].\n"
        "At the start of your response, include the mutation explicitly in the opening sentence.\n"
        "Throughout the response, always refer to this mutation when describing temporal trends, "
        "site differences, and study-level evidence.\n\n"
        "Please write a clear and concise scientific-style summary that follows these instructions:\n"
        "1. **Start with a spatiotemporal overview**: Summarize how prevalence of this mutation "
        "changes over time (using 'yearly') and across sites (using 'by_site') before discussing "
        "any individual studies. Use specific years (e.g., 'from 2013 to 2017') and describe "
        "approximate prevalence levels (e.g., low <1%, moderate 1–10%, high >10%).\n"
        "2. **Temporal trend**: Use 'yearly' to describe increases, decreases, or stability "
        "in mean_prevalence across years. Mention sample sizes when relevant.\n"
        "3. **Spatial differences**: For each site you mention, ALWAYS include the country "
        "by using the 'country' field in data_summary['by_site']. Write sites as '<site> (<country>)'.\n"
        "4. **Study-level consistency**: Use 'by_study' **only as supporting examples**. "
        "Cite studies briefly using 'Authors year_pub' to indicate whether findings across "
        "studies are consistent or divergent. Do **not** focus on a single study if many exist.\n"
        "5. **Caveats**: State any limitations visible in the data—uneven site coverage, "
        "small sample sizes, missing years, or short time series.\n"
        "6. **No invention**: Base all statements **only** on values present in the JSON. "
        "Do not invent additional years, sites, or studies.\n"
    )


    answer = config.call_llm(system_msg, user_msg)
    return answer