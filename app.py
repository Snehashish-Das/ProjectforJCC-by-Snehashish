import nltk
import pickle
from flask import Flask, render_template, request, url_for
from preprocess import preprocess_text
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Download 'punkt' if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

app = Flask(__name__)

# Load the sentiment analysis model and TF-IDF vectorizer
with open('svm.pkl', 'rb') as f:
    svm = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def analyze_sentiment():
    sentiment = None
    boundary_color = '#25252b'  # Default color
    movie_name = None  # Initialize movie_name here
    lottie_path = None  # Initialize lottie_path here
    if request.method == 'POST':
        movie_name = request.form.get('movie_name')
        comment = request.form.get('review')

        # Preprocess the comment
        preprocessed_comment = preprocess_text(comment)

        # Transform the preprocessed comment into a feature vector
        comment_vector = tfidf.transform([preprocessed_comment])

        # Predict the sentiment
        sentiment = svm.predict(comment_vector)[0]

        # Determine the boundary glow color based on sentiment
        boundary_color = 'cyan' if sentiment == 1 else 'red'

        # Determine the Lottie animation path based on sentiment
        lottie_path = 'https://fonts.gstatic.com/s/e/notoemoji/latest/1f929/lottie.json' if sentiment == 1 else 'https://fonts.gstatic.com/s/e/notoemoji/latest/1f631/lottie.json'

    return render_template('index.html', sentiment=sentiment, boundary_color=boundary_color, movie_name=movie_name, lottie_path=lottie_path)

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True)
