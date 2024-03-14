import streamlit as st
# from streamlit.web.cli import main
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration, PegasusTokenizer, PegasusForConditionalGeneration, ProphetNetTokenizer, ProphetNetForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

st.set_option('deprecation.showPyplotGlobalUse', False)

def scrape_links(keyword):
    # Initialize Chrome WebDriver
    service = ChromeService()
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=options)
    driver.get("https://google.com")

    # Websites to search along with Wikipedia
    websites = [
        "wikipedia",
    ]

    scraped_data = {}

    for site in websites:
        # Open Google search with keyword + site
        url = f"https://www.google.com/search?q={keyword}+{site}"
        driver.get(url)
        time.sleep(2)  # Waiting for results to load

        # Get the URL of the first search result
        try:
            link = driver.find_element(By.CSS_SELECTOR, "#search .g a")
            site_link = link.get_attribute('href')
        except:
            continue  # If no link found for this site, move to the next one

        # Navigate to the website page
        driver.get(site_link)
        time.sleep(2)  # Waiting for page to load
        content = driver.page_source
        soup = BeautifulSoup(content, 'html.parser')

        # Get all paragraphs
        paragraphs = soup.find_all('p')

        # Scraping words from paragraphs until at least 1500 words are scraped
        words = []
        for paragraph in paragraphs:
            words.extend(paragraph.get_text().split())
            if len(words) >= 1500:
                break

        # Concatenate words until we have at least 1500 words
        result = ' '.join(words[:1500])
        scraped_data[site] = result

    driver.quit()
    return scraped_data

def generate_abstractive_summary_bart(text):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    output = model.generate(inputs, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary
    # (same as your original generate_abstractive_summary_bart function)

def generate_abstractive_summary_pegasus(text):
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-cnn_dailymail")
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    output = model.generate(inputs, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary
    # (same as your original generate_abstractive_summary_pegasus function)

def generate_summary_prophetnet(text):
    model_name = "microsoft/prophetnet-large-uncased"
    tokenizer = ProphetNetTokenizer.from_pretrained(model_name)
    model = ProphetNetForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(**inputs)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
    # (same as your original generate_summary_prophetnet function)

def generate_abstractive_summary_t5(model_name, text):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    output = model.generate(inputs, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary
    # (same as your original generate_abstractive_summary_t5 function)

def generate_abstractive_summary_falconai(text):
    summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    summary = summarizer(text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return summary[0]['summary_text']
    # (same as your original generate_abstractive_summary_falconai function)

def calculate_scores(summaries, reference_summary):
    vectorizer = CountVectorizer().fit_transform(summaries)
    vectors = vectorizer.toarray()

    cosine_similarities = cosine_similarity(vectors, vectors)

    bleu_scores = [sentence_bleu([reference_summary.split()], summary.split()) for summary in summaries]

    rouge = Rouge()
    rouge_scores = [rouge.get_scores(summary, reference_summary)[0]['rouge-l']['f'] for summary in summaries]

    return cosine_similarities, bleu_scores, rouge_scores
    # (same as your original calculate_scores function)

def main():
    st.title("Summary Comparison App")

    keyword = st.text_input("Enter a keyword:")
    scraped_data = scrape_links(keyword)

    # Generate summaries for each model using the scraped data
    bart_summary = generate_abstractive_summary_bart(scraped_data['wikipedia'])
    pegasus_summary = generate_abstractive_summary_pegasus(scraped_data['wikipedia'])
    prophetnet_summary = generate_summary_prophetnet(scraped_data['wikipedia'])
    t5_large_summary = generate_abstractive_summary_t5("t5-large", scraped_data['wikipedia'])
    t5_small_summary = generate_abstractive_summary_t5("t5-small", scraped_data['wikipedia'])
    t5_base_summary = generate_abstractive_summary_t5("t5-base", scraped_data['wikipedia'])
    falconai_summary = generate_abstractive_summary_falconai(scraped_data['wikipedia'])

    # Create a dictionary with model names as keys and summaries as values
    summaries_dict = {
        "BART": bart_summary,
        "Pegasus": pegasus_summary,
        "ProphetNet": prophetnet_summary,
        "T5_large": t5_large_summary,
        "T5_small": t5_small_summary,
        "T5_base": t5_base_summary,
        "FalconAI": falconai_summary
    }

    reference_summary = st.text_area("Reference Summary")

    if st.button("Submit"):
        cosine_similarities, bleu_scores, rouge_scores = calculate_scores(list(summaries_dict.values()), reference_summary)

        average_cosine_similarity = sum(sum(similarity) for similarity in cosine_similarities) / len(summaries_dict)
        average_bleu_score = sum(bleu_scores) / len(summaries_dict)
        average_rouge_score = sum(rouge_scores) / len(summaries_dict)

        best_summary_index = [average_cosine_similarity, average_bleu_score, average_rouge_score].index(max([average_cosine_similarity, average_bleu_score, average_rouge_score]))
        best_summary_model = list(summaries_dict.keys())[best_summary_index]
        best_summary = summaries_dict[best_summary_model]

        # Display results
        st.subheader("Summary Scores:")
        st.write("Cosine Similarities:", cosine_similarities)
        st.write("BLEU Scores:", bleu_scores)
        st.write("ROUGE Scores:", rouge_scores)

        st.subheader("Average Scores:")
        st.write("Average Cosine Similarity:", average_cosine_similarity)
        st.write("Average BLEU Score:", average_bleu_score)
        st.write("Average ROUGE Score:", average_rouge_score)

        st.subheader("Best Summary Model:")
        st.write(best_summary_model)

        st.subheader("Best Summary:")
        st.write(best_summary)

        # Plot bar graphs
        models = list(summaries_dict.keys())

        # Cosine Similarities
        plt.figure(figsize=(10, 5))
        plt.bar(models, cosine_similarities.mean(axis=1))
        plt.title("Cosine Similarities")
        plt.xlabel("Models")
        plt.ylabel("Cosine Similarity Score")
        st.pyplot()

        # BLEU Scores
        plt.figure(figsize=(10, 5))
        plt.bar(models, bleu_scores)
        plt.title("BLEU Scores")
        plt.xlabel("Models")
        plt.ylabel("BLEU Score")
        st.pyplot()

        # ROUGE Scores
        plt.figure(figsize=(10, 5))
        plt.bar(models, rouge_scores)
        plt.title("ROUGE Scores")
        plt.xlabel("Models")
        plt.ylabel("ROUGE Score")
        st.pyplot()

if __name__ == "__main__":
    main()
