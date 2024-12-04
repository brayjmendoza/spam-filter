# Spam Detector

 This spam detection program has two approaches to predict whether or not a given text is spam.

The first approach uses scikit-learn's Count Vectorizer to create word embellishments 
and then uses a logistic classifier to predict whether an input text is spam or not.
    
The second approach uses Keras's Tokenizer to vectorize the text. This text is then
used to train a neural network that utilizes a LSTM network.

Data taken from Kaggle's Spam Detection Dataset
