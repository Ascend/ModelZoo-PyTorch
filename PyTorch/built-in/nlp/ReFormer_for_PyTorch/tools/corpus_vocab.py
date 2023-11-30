import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm


def process_text(text):
    tokens = word_tokenize(text.lower())

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

    return " ".join(filtered_tokens)


def generate_corpus_and_vocab(csv_file, corpus_file):
    df = pd.read_csv(csv_file)

    corpus_set = set()
    vocab_set = set()

    with open(corpus_file, 'w', encoding='utf-8') as corpus_f:
        for text in tqdm(df['text']):
            processed_text = process_text(text)
            corpus_set.add(processed_text)
            vocab_set.update(processed_text.split())
        
        for text in tqdm(corpus_set):
            corpus_f.write(text + '\n')

if __name__ == '__main__':
    for mode in ["train", "test", "debug"]:
        csv_file = "./datasets/%s.csv" % mode
        corpus_file = "./datasets/%s_corpus.txt" % mode

        generate_corpus_and_vocab(csv_file, corpus_file)
        