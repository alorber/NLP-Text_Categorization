# Andrew Lorber
# NLP Project 1 - Text Categorization

# Imports
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize
from math import log, inf
import numpy as np

# NLTK Installs
# nltk.download('stopwords')
# nltk.download('punkt')


# ------------------------------
# File Pre-Processing Functions
# ------------------------------

# Gets files from user and transforms them into usable lists
def process_input_files():
    # Gets test & train file names
    # training_list_file_name = input("Please enter the name of the training file list.\n")
    # testing_list_file_name = input("Please enter the name of the testing file list.\n")
    # Hardcoded for now
    training_list_file_name = "../Corpus_Sets/corpus1_train.labels"
    testing_list_file_name = "../Corpus_Sets/corpus1_test.list"

    # Creates list of training docs
    training_list_file = open(training_list_file_name, 'r')
    relative_training_path = "/".join(training_list_file_name.split("/")[:-1])  # Relative path of training corpus
    training_doc_list = filter(lambda line: line != "",
                               training_list_file.read().split("\n"))  # List of non-empty lines
    training_doc_list = map(lambda line: relative_training_path + line[1:],
                            training_doc_list)  # Swaps '.' with relative path
    training_doc_list = list(map(lambda line: line.split(" "), training_doc_list))  # Splits line into path & category
    training_list_file.close()

    # Create list of testing docs
    testing_list_file = open(testing_list_file_name, 'r')
    relative_testing_path = "/".join(testing_list_file_name.split("/")[:-1])  # Relative path of testing corpus
    testing_doc_list = list(
        filter(lambda line: line != "", testing_list_file.read().split("\n")))  # List of non-empty lines
    rel_testing_doc_list = list(
        map(lambda line: relative_testing_path + line[1:], testing_doc_list))  # Swaps '.' with relative path
    testing_list_file.close()

    return training_doc_list, testing_doc_list, rel_testing_doc_list


# -------------------
# Training Functions
# -------------------

# Gets training documents statistics and fills dictionaries
def get_doc_stats(doc_list):
    # Dictionaries
    docs_per_category = {}  # Number of docs per category
    token_cnt_per_category = {}  # Token frequency per category {category -> {token -> int} }
    total_tokens_per_category = {}  # Number of tokens per category {category -> int}
    vocabulary = set()  # All unique tokens

    # Loops through training documents
    for [doc_path, doc_category] in doc_list:
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
        tokenized_doc = [word.lower() for word in tokenized_doc if word not in stop_words and word.isalpha()]

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

        doc_file.close()

    # Gets list of categories
    category_list = docs_per_category.keys()
    # Number of training documents
    num_training_docs = sum(docs_per_category.values())

    return (docs_per_category, token_cnt_per_category, total_tokens_per_category,
            vocabulary, category_list, num_training_docs)


# Calculates prior and conditional probabilities
def get_probabilities(docs_per_category, token_cnt_per_category, total_tokens_per_category,
                      vocabulary, category_list, num_training_docs, smoothing_constant):
    category_priors = {}  # P(c) = # doc in c / # doc
    token_category_conditional = {}  # P(t|c) = # token t in c / total # tokens in c
                                     # { category -> {token -> P(t|c)} }

    # Loops through categories
    for category in category_list:
        # Calculates prior
        category_priors[category] = docs_per_category[category] / num_training_docs

        # Calculates conditional probability
        token_category_conditional[category] = {}
        denominator = total_tokens_per_category[category] + (smoothing_constant * len(vocabulary))
        for token in vocabulary:
            # Checks if token has been seen in category
            if token in token_cnt_per_category[category].keys():
                token_category_conditional[category][token] = token_cnt_per_category[category][token] / denominator
            else:
                token_category_conditional[category][token] = smoothing_constant / denominator

    return category_priors, token_category_conditional


# ------------------
# Testing Functions
# ------------------

# Gets category predictions for testing documents
def get_predictions(rel_testing_doc_list, category_priors, token_category_conditional, vocabulary, category_list):
    predictions = []  # List of predictions

    # Loops through testing documents
    for doc_path in rel_testing_doc_list:
        # Gets document statistics
        # -------------------------

        doc_token_cnt = {}  # Frequency of tokens in current document

        # Opens doc
        doc_file = open(doc_path, 'r')

        # Tokenize doc
        tokenized_doc = word_tokenize(doc_file.read())

        # Removes stop words
        tokenized_doc = [word.lower() for word in tokenized_doc if word not in stop_words and word.isalnum()]

        # Loops through words in doc
        for token in tokenized_doc:
            # Stems token
            stemmed_token = Stemmer.stem(token)

            # Updates token frequency in doc
            if stemmed_token not in doc_token_cnt:
                doc_token_cnt[stemmed_token] = 1
            else:
                doc_token_cnt[stemmed_token] += 1

        doc_file.close()

        # Determines Category
        # --------------------

        prediction = ("", -inf)  # Stores most probable category: (category, probability)

        # Loops through categories
        for category in category_list:
            prior_prob = log(category_priors[category])
            conditional_prob = 0

            # Calculates conditional probability for each token in doc
            for token in doc_token_cnt.keys():
                if token in vocabulary:
                    conditional_prob += log(token_category_conditional[category][token]) * doc_token_cnt[token]

            # Checks if category
            prob = prior_prob + conditional_prob
            if prob > prediction[1]:
                prediction = (category, prob)

        # Stores prediction
        predictions.append(prediction[0])

    return predictions


# Writes predictions to output file
def write_to_output(testing_doc_list, predictions):
    # Write to output file
    # --------------------

    # Gets output file name
    # out_file_name = input("Please enter the name of the output file.\n")
    out_file_name = "./out.labels"  # Hard-coded for now

    # Opens output file
    out_file = open(out_file_name, 'w')

    # Writes predictions to output file
    for (doc_path, prediction) in zip(testing_doc_list, predictions):
        # out_file.write(f"{doc_path} {prediction}\n")
        out_file.write(doc_path + " " + prediction + '\n')


# Calculates accuracy of predictions
def calculate_accuracy(predictions):
    # Calculates accuracy
    test_answers_file_name = "../Corpus_Sets/corpus1_test.labels"
    test_answers_file = open(test_answers_file_name, 'r')
    test_answers_list = filter(lambda line: line != "", test_answers_file.read().split("\n"))  # List of non-empty lines
    test_answers_list = list(map(lambda line: line.split(" ")[1], test_answers_list))
    test_answers_file.close()

    correct = 0
    incorrect = 0
    for (prediction, actual) in zip(predictions, test_answers_list):
        if prediction == actual:
            correct += 1
        else:
            incorrect += 1

    accuracy = correct / (correct + incorrect)

    # print(f"Correct: {correct}\nIncorrect: {incorrect}\nAccuracy: {accuracy}\n")

    return accuracy


# Finds best smoothing constant
def get_best_smoothing_constant(rel_test_doc_list, doc_stats):
    best = []  # Top 10: (smoothing constant, accuracy)
    for smoothing_constant in np.arange(.01, .91, .01):
        # Calculates probabilities
        prior_probabilities, conditional_probabilities = get_probabilities(*doc_stats, smoothing_constant)

        # Gets test documents predictions
        prediction_list = get_predictions(rel_test_doc_list, prior_probabilities, conditional_probabilities,
                                          doc_stats[3], doc_stats[4])

        # Calculates Accuracy on Corpus 1
        acc = calculate_accuracy(prediction_list)

        # Checks top 10
        best.append((smoothing_constant, acc))  # Adds to list
        if len(best) > 10:
            best.sort(key=lambda el: el[1], reverse=True)  # Sorts list
            best.pop()

    # Prints top 10
    best.sort(key=lambda el: el[1], reverse=True)  # Sorts list
    for num, item in enumerate(best, start=1):
        print(f"{num}. Smoothing Constant: {item[0]}  -  Accuracy: {item[1]}\n")

    return best


# -----
# MAIN
# -----

# Constants
stop_words = set(stopwords.words('english'))
Stemmer = PorterStemmer()
# smoothing_constant = .07

# Processes input files
train_doc_list, test_doc_list, rel_test_doc_list = process_input_files()

# Gets training documents statistics
stats = get_doc_stats(train_doc_list)

get_best_smoothing_constant(rel_test_doc_list, stats)

'''
# Calculates probabilities
prior_probabilities, conditional_probabilities = get_probabilities(*stats)

# Gets test documents predictions
prediction_list = get_predictions(rel_test_doc_list, prior_probabilities, conditional_probabilities, stats[3], stats[4])

# Writes predictions to output file
write_to_output(test_doc_list, prediction_list)

# Calculates Accuracy on Corpus 1
calculate_accuracy(prediction_list)
'''



# Todo: Things to test
#   - Convert to lower case
#   - Stop words COMPLETE
#   - Lemmatization
#   - Different smoothing methods
#   - K folds
#   - Don't include punctuation
#   - Log probabilities
#   - Move tokenization to function

