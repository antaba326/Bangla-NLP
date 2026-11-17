import re

def clean_text(text):
    """
    Clean Bangla text before processing.
    """
    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()
