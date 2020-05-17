import pandas as pd
from flask import request, Flask ,render_template

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
# Importing the dataset
    dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
    import re
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    corpus = []
    for i in range(0, 1000):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    
    # Creating the Bag of Words model
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 1500)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 1].values
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    
    # Fitting RANDOM FOREST to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    
    if request.method == 'POST':
        print("I started")
        message = request.form['message']
        data = [message]
        print(data)
        data = re.sub('[^a-zA-Z]', ' ',str(data))
        data = data.lower()
        data = data.split()
        print('Splitted ; ' , data)
        ps = PorterStemmer()
        data = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        data = ' '.join(data)
        print('Joined :  ' , data)
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        print(my_prediction)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True, use_reloader = False)

