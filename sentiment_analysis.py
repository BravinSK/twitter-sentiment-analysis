import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import string
import re

# Load the dataset
df = pd.read_csv('../twitter.csv', encoding='latin-1', header=None)
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Display the first 5 rows
print(df.head())

# Check the distribution of sentiments
print(df['target'].value_counts())

# To download NLTK stopwords and punctuation
nltk.download('stopwords')
nltk.download('punkt')

def clean_text(text):
    
    # Converting uppercase to lowercase
    text = text.lower()

    # Remove the URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove user mentions
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags
    text = re.sub(r'#\w+', '', text)

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply the cleaning function to the dataset
df['cleaned_text'] = df['text'].apply(clean_text)
print(df['cleaned_text'].head())

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the cleaned text
X = vectorizer.fit_transform(df['cleaned_text'])

# Target variable (0 = negative, 4 = positive)
y = df['target'].map({0: 0, 4: 1})  # Convert 4 to 1 for positive sentiment

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Multinomial Naive Bayes algorithm

# Initialize the model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Predict the sentiment for the test set
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

def predict_sentiment(text):

    # Clean the input text
    cleaned_text = clean_text(text)

    # Transform the text into features
    text_vector = vectorizer.transform([cleaned_text])

    # Predict sentiment
    prediction = model.predict(text_vector)
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Test the function
print(predict_sentiment("I love this movie! It's amazing!")) 
print(predict_sentiment("This is the worst experience ever."))  