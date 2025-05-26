"""
Functions for forming a request to LLM, generating a response, and outputting it.

Functions:
- chat_template_groq: creates a message in OpenAI chat format based on the request and relevant chunks.
- client_response_groq: calls LLM via the GROQ API with a built-in dialog.
- print_response_markdown: outputs the LLM response in Markdown format (Streamlit).
- print_response_console: outputs the LLM response in console mode.
"""

from openai import OpenAI
import json
import streamlit as st

from .config import BASE_PROMPT, CHUNKS_PATH, DIALOGUE_INSTRUCTION, LLM_GROQ, QUERY


def chat_template_groq(indices: list[int],
                       query: str = QUERY,
                       base_prompt: str = BASE_PROMPT) -> list[dict]:
    """
    Creates a dialog template in OpenAI chat format.

    Arguments:
    - indices: indices of relevant chunks.
    - query: user query.
    - base_prompt: prompt template with placeholders.

    Returns:
    - a list of dictionaries with "system" and "user" roles for LLM.
    """
    # Save chunks
    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        chunks_and_statistics = json.load(file)
    chunks_texts = [chunk["text"] for chunk in chunks_and_statistics]

    # Form context from relevant chunks
    context = "- " + "\n- ".join(chunks_texts[idx] for idx in indices)

    # Insert context and query into a template
    full_prompt = base_prompt.format(context=context, query=query)

    # Form a dialogue template
    dialogue_template = [
        DIALOGUE_INSTRUCTION,
        {
            "role": "user",
            "content": full_prompt
        }
    ]
    return dialogue_template


def client_response_groq(dialogue_template: list[dict],
                         base_url: str=LLM_GROQ["base_url"],
                         api_key: str=LLM_GROQ["api_key"],
                         model=LLM_GROQ["model"]) -> str:
    """
    Makes a request to LLM via the GROQ API with a dialogue.

    Arguments:
    - dialogue_template: a list of messages for the model.
    - base_url: GROQ API address.
    - api_key: API access key.
    - model: model name.

    Returns:
    - the model response text.
    """
    if not api_key:
        raise ValueError("API key is required for InferenceClient.")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    response = client.chat.completions.create(
        model=model,
        messages=dialogue_template
    )
    response_text = response.choices[0].message.content
    
    return response_text


def print_response_markdown(response_text: str):
    """
    Outputs the LLM response in Streamlit as Markdown.
    If the response is JSON with an "answer" field, output it. Otherwise, output the raw text.
    """
    try:
        parsed = json.loads(response_text)
        answer = parsed.get("answer", "No answer field found in the response.")
        st.markdown(answer)
    except Exception:
        st.markdown(response_text)


def print_response_console(response_text: str):
    """
    Prints the LLM response to the console.
    If the response is JSON with the "answer" field, prints it. Otherwise, prints the raw text.
    """
    try:
        # Parse json
        parsed = json.loads(response_text)
        answer = parsed.get("answer", "No answer field found in the response.")
        print(answer)
    except Exception:
        # If not - print as it is
        print(response_text)
