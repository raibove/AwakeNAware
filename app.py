from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from nltk.corpus import stopwords
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from textblob import Word 
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	Suicide = pd.read_csv('suicide.csv',encoding='ISO-8859-1')
	Suicide['Tweet'] = Suicide['Tweet'].fillna("")
	Suicide['lower_case']= Suicide['Tweet'].apply(lambda x: x.lower())      
	
	tokenizer = RegexpTokenizer(r'\w+')
	Suicide['Special_word'] = Suicide.apply(lambda row: tokenizer.tokenize(row['lower_case']), axis=1)    

	freq = pd.Series(' '.join(Suicide['Tweet']).split()).value_counts()[-10:]                       
	freq = list(freq.index)
	Suicide['Contents'] = Suicide['Tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq)) 

	stop = stopwords.words('english')
	Suicide['stop_words'] = Suicide['Special_word'].apply(lambda x: [item for item in x if item not in stop])  

	Suicide['stop_words'] = Suicide['stop_words'].astype('str')
	Suicide['short_word'] = Suicide['stop_words'].str.findall('\w{3,}')         
	Suicide['string'] =Suicide['stop_words'].replace({"'": '', ',': ''}, regex=True)
	Suicide['string'] = Suicide['string'].str.findall('\w{3,}').str.join(' ') 

	nltk.download('words')
	words = set(nltk.corpus.words.words())
	Suicide['NonEnglish'] = Suicide['string'].apply(lambda x: " ".join(x for x in x.split() if x in words))  

	Suicide['tweet'] = Suicide['NonEnglish'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 

	Suicide['label'] = Suicide['Suicide'].map({'Potential Suicide post':0,'Not Suicide post':1})
	X = Suicide['tweet']
	y = Suicide['label']
	cv = CountVectorizer()
	X = cv.fit_transform(X)

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	y_train = np.nan_to_num(y_train)
	y_test = np.nan_to_num(y_test)
	model = MultinomialNB()
	model.fit(X_train,y_train)
	model.score(X_test,y_test)
	
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = model.predict(vect)
	return render_template('result.html',prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)