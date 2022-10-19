''''
train_classifier.py
This file should take in a database file name from the command lines and convert database
into a data frame which produces a machine leanring model that will be saved as
as pickle file.
'''
import sys,re
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputClassifier

#For tfidf

class FormatTextTransformer(BaseEstimator, TransformerMixin):
#Class for formatting the text
	def fit(self, X, y=None):
		return self
	def transform(self, X, y=None):	
		'''
		Input
		X: Dataframe
		Y: None
		Output:
		message_extract: String ready to go through TFIDF
		'''
		def tokenize(text):
			wxd = re.sub(r'[^\w\s]','',str(text.lower()))
			from nltk.corpus import stopwords
			from nltk.stem import WordNetLemmatizer
			lemmatizer = WordNetLemmatizer()
			from nltk.tokenize import word_tokenize
			sr= stopwords.words('english')
			wxd = word_tokenize(wxd)
			wxd = [w for w in wxd if not w in sr]

			holdlist = ""
			for i in wxd:
				holdlist += lemmatizer.lemmatize(i)
				holdlist +=  " "
			return holdlist
		# Perform arbitary transformation
		message_extract = X.apply(tokenize)
		return message_extract

def load_data(database_filepath):
	'''
	Input:
	database_filepath : path from the command line to the database file
	Output:
	db_table: Dataframe of loaded table
	'''
	import sqlalchemy as db
	sqlfp = 'sqlite:///' + database_filepath
	engine = db.create_engine(sqlfp)
	connection = engine.connect()
	metadata = db.MetaData()
	dp_table = "This is not a database please try again"
	#Check if the file name is a db file which can connect for table loading
	if sqlfp[-3:] == '.db':
		table_name = database_filepath.split('/')[-1]
		dp_table = pd.read_sql_table(table_name[:-3], connection)
	connection.close()
	engine.dispose()
	return dp_table
	
def build_model():
	'''
	Builds the model
	Input:
	None
	Output
	grid_pipe: GridSearchCV pipe
	'''
	radius_params={'xgb__estimator__n_estimators':[2,8], \
		'xgb__estimator__max_depth': [5,15],\
		'xgb__estimator__objective': [ 'binary:logistic' ],
		'xgb__estimator__eval_metric': ['logloss'],
		'xgb__estimator__learning_rate': [0.5],
		'xgb__estimator__use_label_encoder': [False]}
#	Pipeline will do the following, clean the text, transform via tfidf, 
#and then do radius neighbors classification
	cls = Pipeline([
	('format_text', FormatTextTransformer()),
# 	('label', LabelEncoder()),
    ('tfidf', TfidfVectorizer()),
   ('xgb',MultiOutputClassifier(XGBClassifier()))
	])
#	Build the Gridsearch CV pipe
	grid_pipe = GridSearchCV(cls,
		scoring='accuracy',
		param_grid=radius_params,
		verbose=1)
	return grid_pipe
	
def evaluate_model(model, X_test, Y_test, category_names=None):
#Evaluates the model
	return accuracy_score(Y_test, model.predict(X_test))


def save_model(model, model_filepath):
#Saves the model
	import pickle
	with open(model_filepath, 'wb') as f:
		pickle.dump(model, f)
	return True


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        df = load_data(database_filepath)
        df = df.fillna(0)
        X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:,0], df.iloc[:,1:], test_size=0.2, random_state=4345  )
        
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