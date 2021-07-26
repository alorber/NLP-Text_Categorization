###### ECE467 (Natural Language Processing) Project 1
# Text Categorization

The goal of this project was to implement a text categorization system using one of the machine learning methods discussed in class.

The program allows the user to specify the names of two input files. The first contains a list of labeled training documents. The program uses these training documents to train itself appropriately so that it can predict the labels of future documents according to the learned categories. The second file contains a list of unlabeled test documents. After all of the predictions have been made, the program asks the user to specify the name of one output file to print results to.

For my implementation, I chose to use the Naïve Bayes machine learning method. 

The program tokenizes documents using the punkt nltk word-tokenizer. The tokens are then filtered by removing words present in the nltk stop-word list and words that do not consist of letters (i.e. punctuation). Next, the words are stemmed using the Porter Stemmer.
Laplace Smoothing was used for the smoothing method; To find the best smoothing constant, the program was run on all values between 0.01 and 0.9 inclusive with a 0.01 step size. The constant that caused the best average results was found and hardcoded into the program. I considered using k-fold validation on the training set to determine the best smoothing constant at run-time for any given corpus, but was unsure if that would be allowed. 

Different parameters experimented with include: case sensitivity, stop-words, smoothing constant, and filtering tokens by characters. Converting each word to lowercase appeared to make the accuracy slightly worse. If this program were to be used in production, the time and space saved would probably be worth the negligible loss in accuracy, but I decided to go with the slightly higher accuracy. Removing the stop words raised the accuracy by a small amount. Removing punctuation and counting words consisting of letters appeared to have no negative affect on the program, so it was implemented.
