from flask import Flask, render_template, request
from botConfig import myBotName, chatBG, botAvatar
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn import metrics

application = Flask(__name__)
chatbotName = myBotName
print("Bot Name set to: " + chatbotName)
print("Background is " + chatBG)
print("Avatar is " + botAvatar)

# Load data from CSV file
data = pd.read_csv('Datasets/Final_PreProcessed_Dataset.csv')

# Preprocess the data
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_questions(questions):
    # Tokenize questions
    words = word_tokenize(questions.lower())
    # Remove stop words
    words = [w for w in words if not w in stop_words]
    # Stem words
    words = [ps.stem(w) for w in words]
    # Join words back into a string
    return ' '.join(words)

# Apply preprocessing to all questionss in the data
data['questions'] = data['questions'].apply(preprocess_questions)

# Train the chatbot
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Create a bag-of-words representation of the questionss
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['questions'])
y = data['answers']

# Train a Naive Bayes classifier
clf = MultinomialNB()
model = clf.fit(X, y)

# Define a function to get the chatbot answers
def get_answers(questions):
    # Preprocess the questions
    questions = preprocess_questions(questions)
    # Convert the questions to a bag-of-words representation
    X_test = vectorizer.transform([questions])
    # Get the predicted answers from the classifier
    answers = clf.predict(X_test)[0]
    return answers

@application.route("/")
def home():
    return render_template("index.html", botName = chatbotName, chatBG = chatBG, botAvatar = botAvatar)

@application.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    botReply = get_answers(userText)
    return botReply


if __name__ == "__main__":
    #application.run()
    pred = model.predict(X)
    print(metrics.accuracy_score(y,pred))
    application.run(host='0.0.0.0', port=8080)
    
