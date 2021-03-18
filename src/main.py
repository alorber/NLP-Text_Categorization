# Andrew Lorber
# NLP Project 1 - Text Categorization

# Imports
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize
from math import log

# NLTK Installs
# nltk.download('stopwords')
# nltk.download('punkt')

stop_words = set(stopwords.words('english'))
Stemmer = PorterStemmer()

# --------------------
# File Pre-Processing
# --------------------

# Gets test & train file names
# training_list_file_name = input("Please enter the name of the training file list.\n")
# testing_list_file_name = input("Please enter the name of the testing file list.\n")
# Hardcoded for now
training_list_file_name = "../Corpus_Sets/corpus1_train.labels"
testing_list_file_name = "../Corpus_Sets/corpus1_test.labels"

# Creates list of training docs
training_list_file = open(training_list_file_name, 'r')
relative_training_path = "/".join(training_list_file_name.split("/")[:-1])  # Relative path of training corpus
training_doc_list = filter(lambda line: line != "", training_list_file.read().split("\n"))   # List of non-empty lines
training_doc_list = map(lambda doc: relative_training_path + doc[1:], training_doc_list)  # Swaps '.' with relative path
training_doc_list = list(map(lambda doc: doc.split(" "), training_doc_list))  # Splits line into path & category

# Create list of testing docs
testing_list_file = open(testing_list_file_name, 'r')
relative_testing_path = "/".join(testing_list_file_name.split("/")[:-1])  # Relative path of testing corpus
testing_doc_list = testing_list_file.read().split("\n")   # List of lines
testing_doc_list = map(lambda doc: relative_testing_path + doc[1:], testing_doc_list)  # Swaps '.' with relative path

# ---------
# Training
# ---------

# Document Statistics Logging
# ----------------------------

# Dictionaries
docs_per_category = {}  # Number of docs per category
token_cnt_per_category = {}  # Token frequency per category {category -> {token -> int} }
total_tokens_per_category = {}  # Number of tokens per category {category -> int}
vocabulary = set()  # All unique tokens

# Loops through training documents
for [doc_path, doc_category] in training_doc_list:
    # Logs category
    if doc_category in docs_per_category:
        docs_per_category[doc_category] += 1
    else:
        docs_per_category[doc_category] = 1

    # Opens doc
    doc_file = open(doc_path, 'r')

    # Tokenize doc
    tokenized_doc = word_tokenize(doc_file.read())

    # Removes stop words
    tokenized_doc = [word for word in tokenized_doc if word not in stop_words]

    # Loops through words in doc
    for token in tokenized_doc:
        # Stems token
        stemmed_token = Stemmer.stem(token)

        # Updates token frequency in category
        if doc_category not in token_cnt_per_category:
            token_cnt_per_category[doc_category] = {}
        if stemmed_token in token_cnt_per_category[doc_category]:
            token_cnt_per_category[doc_category][stemmed_token] += 1
        else:
            token_cnt_per_category[doc_category][stemmed_token] = 1

        # Updates total token count per category
        if doc_category in total_tokens_per_category:
            total_tokens_per_category[doc_category] += 1
        else:
            total_tokens_per_category[doc_category] = 1

        # Adds token to vocabulary
        vocabulary.add(stemmed_token)

# Gets list of categories
category_list = docs_per_category.keys()
# Number of training documents
num_training_docs = sum(docs_per_category.values())

# Probability Calculations
# -------------------------

category_priors = {}  # P(c) = # doc in c / # doc
token_category_conditional = {}  # P(t|c) = # token t in c / total # tokens in c
                                 # { category -> {token -> P(t|c)} }

# Loops through categories
for category in category_list:
    # Calculates prior
    category_priors[category] = docs_per_category[category] / num_training_docs

    # Calculates conditional probability
    token_category_conditional[category] = {}
    for token in vocabulary:
        # Checks if token has been seen in category
        if token in token_cnt_per_category[category].keys():
            token_category_conditional[category][token] = token_cnt_per_category[category][token] \
                                                          / total_tokens_per_category[category]
        else:
            token_cnt_per_category[category][token] = 0  # TODO: Add smoothing


# Gets output file name
# out_file = input("Please enter the name of the output file.\n")


# Todo: Things to test
#   - Convert to lower case
#   - Stop words COMPLETE
#   - Lemmatization
#   - Different smoothing methods
#   - K folds
#   - Don't include punctuation
#   - Log probabilities

