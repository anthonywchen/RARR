"""Launches a demo to run PURR on.

To run:
streamlit run launch_editor_demo.py --server.baseUrlPath rarr --server.fileWatcherType none
"""
import json
import time
from typing import Any, Dict

import streamlit as st

from run_editor_sequential import run_editor_one_instance


def call_rarr(
    claim: str,
    model: str,
    temperature_qgen: float,
    num_rounds_qgen: int,
) -> Dict[str, Any]:
    """Cached call to the RARR function."""
    result = run_editor_one_instance(
        claim=claim,
        model=model,
        temperature_qgen=temperature_qgen,
        num_rounds_qgen=num_rounds_qgen,
    )
    return result


def main() -> None:
    """Launches a streamlit RARR demo"""
    st.set_page_config(layout="wide")
    st.title("GPT-3 RARR Demo")
    st.caption(
        "Uses GPT-3 as the question generator, agreement gater, and "
        "editor, Bing as the search engine, and a small cross-encoder as the passage "
        "extractor."
    )
    st.caption("Written by Anthony Chen. [anthony.chen@uci.edu]")
    model = st.sidebar.selectbox(
        "Choose a GPT-3 model.",
        (
            "text-davinci-003",
            "text-davinci-002",
            "text-davinci-001",
            "text-curie-001",
            "text-davinci-001",
            "text-babbage-001",
            "text-ada-001",
            "code-davinci-002",
            "code-davinci-001",
        ),
    )
    temperature_qgen = st.sidebar.number_input(
        "Temperature for question generation", min_value=0.0, max_value=1.0, value=0.7
    )
    num_rounds_qgen = st.sidebar.number_input(
        "Number of rounds of question generation", min_value=1, max_value=3, value=1
    )
    claim = st.text_input(
        "Enter a claim to edit.", "Michael Jordan was an NFL player with the LA Lakers."
    )

    if st.button("Run Editing"):
        start = time.time()
        result = call_rarr(
            claim=claim,
            model=model,
            temperature_qgen=temperature_qgen,
            num_rounds_qgen=num_rounds_qgen,
        )
        st.write(f"RARR Editing Took {time.time()-start:.2f} Seconds.")
        st.download_button(
            "Download JSON Output", json.dumps(result, ensure_ascii=False)
        )
        st.json(result)


if __name__ == "__main__":
    main()
