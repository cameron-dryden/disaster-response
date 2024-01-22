import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(database_filepath, engine)
    X = df["message"]
    y = df.iloc[:, 4:]

    return X, y


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "urlplaceholder", text)

    words = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = []
    [cleaned_tokens.append(lemmatizer.lemmatize(word).strip()) for word in words]
    
    return cleaned_tokens


def build_model():
    pipeline = Pipeline([
    ("count_vect", CountVectorizer(tokenizer = tokenize)),
    ("tfidf", TfidfTransformer()),
    ("clf", MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return pipeline



def evaluate_model(model, X_test, Y_test):
    y_pred = model.predict(X_test)

    avg_model_score = np.asarray([0, 0, 0], dtype = np.float32)

    for i in range(y_pred.shape[1]):
        report = classification_report(Y_test.iloc[:, i], pd.DataFrame(y_pred).iloc[:, i], output_dict=True, zero_division=0)['weighted avg']
    
        avg_model_score[0] += report['f1-score']
        avg_model_score[1] += report['precision']
        avg_model_score[2] += report['recall']

    avg_model_score = avg_model_score / y_pred.shape[1]
    print("Overall Model Performance:")
    print(f"f1 score: {avg_model_score[0]}, precision: {avg_model_score[1]}, recall: {avg_model_score[2]}")


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()