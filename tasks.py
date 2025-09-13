# text_summarization_tool.py

from transformers import pipeline

def summarize_text(text, max_length=130, min_length=30):
    """
    Summarizes the input text using a transformer-based model.

    Args:
        text (str): The input text to summarize.
        max_length (int): Max length of the summary.
        min_length (int): Min length of the summary.

    Returns:
        str: The summarized text.
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']


if __name__ == "__main__":
    print("====== TEXT SUMMARIZATION TOOL ======\n")
    user_input = input("Enter the article or paragraph to summarize:\n\n")

    print("\nSummarizing...\n")
    summarized_text = summarize_text(user_input)
    print("------ SUMMARY ------\n")
    print(summarized_text)
