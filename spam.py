import numpy as np
import pandas as pd

# Get data for spam detection
df = pd.read_csv("https://raw.githubusercontent.com/AashitaK/datasets/main/spam.csv", encoding="latin-1")

# Modify dataframe, columns will have label, text, text_length
df = df.dropna(how="any", axis=1)
df.columns = ['label', 'text']
df['text_length'] = df['text'].apply(lambda x: len(x.split(' ')))

# Initialize training and testing sets
from sklearn.model_selection import train_test_split
X = df['text'] 
y = df['label'].replace({'spam': 1, 'ham': 0})
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

import re
def cleanText(text):
    """
    Preprocesses a given text for word embeddings.
    
    - Removes HTML tags
    - Removes special characters
    - Puts text in lowercase
    """
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and replace it with a space
    filter = "`~!@#$%^&*()_-+={|}[\\]'\";:,.<>/?\t\n"
    translateDict = {c: " " for c in filter}
    translationTable = str.maketrans(translateDict)
    text = text.translate(translationTable)
    
    # Put text in lowercase
    text = text.lower()

    return text

print("===============LOGISTIC CLASSIFER===============")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Create a vectorizer and train it
vectorizer = CountVectorizer(stop_words="english", preprocessor=cleanText)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_valid_vectorized = vectorizer.transform(X_valid)
                                          
# Create and train a logistic classifier
LR_clf = LogisticRegression()
LR_clf.fit(X_train_vectorized, y_train)

# Evaluation
print('Accuracy on training set: {:.2f}'
     .format(LR_clf.score(X_train_vectorized, y_train) * 100))
print('Accuracy on validation set: {:.2f}'
     .format(LR_clf.score(X_valid_vectorized, y_valid) * 100))


print("\n EVALUATION METRICS:\n")

# Compute confusion matrix
from sklearn.metrics import confusion_matrix
y_predicted = LR_clf.predict(X_valid_vectorized)
confusion = confusion_matrix(y_valid, y_predicted)
print('Confusion Matrix\n', confusion, "\n")

# Compute Accuracy, Precision, Recall, and F1 Score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_valid, y_predicted)
precision = precision_score(y_valid, y_predicted)
recall = recall_score(y_valid, y_predicted)
f1 = f1_score(y_valid, y_predicted)
print('Accuracy:', accuracy) # Not a good metric since dataset is imbalanced
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1) # We want a higher f1 score since the dataset is imbalanced

# Filter spam with a neural network 
print("\n===============NEURAL NETWORK===============")
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

# Vectorize text for neural network
num_words = 50000
tokenizer = Tokenizer(num_words)
tokenizer.fit_on_texts(X)
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_valid_tokens = tokenizer.texts_to_sequences(X_valid)

# Pad text to train neural network
max_tokens = 200
X_train_padded = pad_sequences(X_train_tokens, maxlen=max_tokens)
X_valid_padded = pad_sequences(X_valid_tokens, maxlen=max_tokens)

# Define a model
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.optimizers.legacy import Adam
from keras.losses import SparseCategoricalCrossentropy

# Define model for neural network
model = Sequential()
embedding_size = 16
model.add(Embedding(input_dim=num_words, output_dim=embedding_size))
model.add(Dropout(0.2))
model.add(LSTM(8))
model.add(Dense(units=2, activation="softmax"))

# Build model
model.compile(loss=SparseCategoricalCrossentropy(), optimizer=Adam(), metrics=["accuracy"])

# Train model
model.fit(X_train_padded, y_train, batch_size=16, epochs=5, verbose=1, validation_data=(X_valid_padded, y_valid))

# ===== EVALUATION METRICS =====
print("EVALUATION METRICS:\n")

# Get model's predicted values for the testing set
y_prob = model.predict(X_valid_padded, verbose=0)
y_predicted = y_prob.argmax(axis=-1)

# Compute confusion matrix
confusion = confusion_matrix(y_valid, y_predicted)
print('\nConfusion Matrix\n', confusion, "\n")

# Compute Accuracy, Precision, Recall, and F1 Score
accuracy = accuracy_score(y_valid, y_predicted)
precision = precision_score(y_valid, y_predicted)
recall = recall_score(y_valid, y_predicted)
f1 = f1_score(y_valid, y_predicted)
print('Accuracy:', accuracy) # Not a good metric since dataset is imbalanced
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1) # Want a higher f1 score since the dataset is imbalanced