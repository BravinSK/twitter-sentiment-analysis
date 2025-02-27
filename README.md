# Twitter Sentiment Analysis 

This project performs sentiment analysis on Twitter data using Python and machine learning. It classifies tweets as either **positive** or **negative** using a **Multinomial Naive Bayes** model. The project includes data preprocessing, model training, and a simple Flask-based web interface for real-time predictions.

## Features 
- **Data Cleaning**: Removes URLs, mentions, hashtags, and stopwords.
- **Model Training**: Uses `scikit-learn`'s `MultinomialNB` for sentiment classification.
- **Web Interface**: A Flask-based frontend to input text and get sentiment predictions.
- **Evaluation**: Provides accuracy, classification report, and confusion matrix.

## Technologies Used 🛠️
- **Python**: Core programming language.
- **Pandas**: Data manipulation and analysis.
- **NLTK**: Natural Language Toolkit for text preprocessing.
- **Scikit-learn**: Machine learning model training and evaluation.
- **Flask**: Backend for the web interface.
- **HTML/CSS/JavaScript**: Frontend for the web interface.


## Installation 

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   
6. **Run the Flask App**:
   ```bash
   python app.py

8. **Open the Web Interface**:  
   Visit http://127.0.0.1:5000/ in your browser.
  
   
## Usage 
- Enter a tweet or text in the input box and click "Analyze Sentiment" to see if it's **positive** or **negative**.

## Dataset 
-  ```bash
   https://www.kaggle.com/datasets/kazanova/sentiment140
   
- The dataset used for training is `twitter.csv`, which contains labeled tweets (0 = negative, 4 = positive).

## Results 
- **Accuracy**: ~80% (may vary based on dataset and preprocessing).
- **Classification Report**: Precision, recall, and F1-score for both classes.
- **Confusion Matrix**: Visual representation of model performance.

## Contributing 
Feel free to contribute to this project! Open an issue or submit a pull request.
