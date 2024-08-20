import requests
import re
from load_config import LoadConfig
CFG = LoadConfig()


def load_swiss_faq():
    """
    Fetches and processes the Swiss FAQ document from a remote URL.

    This function performs the following steps:
    1. Retrieves the FAQ text from a URL specified in the configuration.
    2. Splits the FAQ text into separate sections based on the section headers.
    3. Returns the processed FAQ sections as a list of dictionaries, where each dictionary
       contains the text of one section.

    The URL for the FAQ document is obtained from the `CFG.swiss_faq_url` configuration setting.

    Returns:
        list[dict]: A list of dictionaries where each dictionary represents a section of the FAQ. 
        Each dictionary has a single key:
            - "page_content": The text content of the FAQ section.

    Raises:
        requests.HTTPError: If the request to fetch the FAQ document fails.
        ValueError: If the FAQ content cannot be parsed correctly.
    """

    # Fetch the FAQ text from a remote URL
    response = requests.get(
        CFG.swiss_faq_url
    )
    response.raise_for_status()
    faq_text = response.text

    # Split the FAQ text into separate documents based on section headers
    docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]
    return docs
