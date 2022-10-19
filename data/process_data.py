''''
ProcessData.py
This function should take in two files from the data folder of message and categories files
merge them together and produce one larger file in a SQLite file.
'''
import sys


def load_data(messages_filepath, categories_filepath):
	'''
	Input:
	message_filepath: sting of message filepath
	categories_filepath: sting of categories filepath
	returns: 
	df_final_2: Dataframe
	'''
	import pandas as pd
	messages = pd.read_csv(messages_filepath)
	categories = pd.read_csv(categories_filepath)
	data = messages.merge(categories, on='id')
	return data


def clean_data(df):
	'''
	Input:
	df: Dataframe
	returns: 
	df_final_2: Dataframe
	'''
	import pandas as pd
	import numpy as np
# 	column_names = df['categories'][0]
# 	#Get tokenizer for column names
	from nltk.tokenize import RegexpTokenizer
# 	tokenizer = RegexpTokenizer("-[0-1][;]", gaps=True)
# 	column_names = tokenizer.tokenize(column_names)
# 	column_names[-1] = column_names[-1].replace('-0','')
	#create new tokenizer and parse data
	tokenizer = RegexpTokenizer("[;]", gaps=True)
#Feature cross	
	df['categories'] = df[['categories','genre']].apply(lambda x: x[0].replace('-',\
	'_' + str(x[1]) + '-'),axis=1)
	category_list = df['categories'].apply(tokenizer.tokenize)
# 	#Turn category list within data series into dataframe
	category_df = category_list.apply(pd.Series)
# 	#Turn dataframe into binary codes
	category_df2 = category_df.applymap(lambda x: x.replace('-1','') if int(x[-1]) ==1
		else np.nan)
# # 	category_df2.columns = column_names
# 	#Combine sets to make data frame
	category_df2 = pd.get_dummies(category_df2, prefix="", prefix_sep="")
	categories_clean = pd.concat([df, category_df2],axis=1)
# 	df_final = pd.get_dummies(categories_clean, columns=['genre'])
	df_final_2 = categories_clean.drop(['categories','genre','id','original'],axis=1).fillna(0)
	return df_final_2

def save_data(df, database_filename):
	'''
	Input:
	df: Dataframe
	database_Filename: sting of database filepath and name
	returns: 
	String: File saved
	'''
	from sqlalchemy import create_engine
	sql_path = "sqlite:///" + database_filename
	engine = create_engine(sql_path, echo=False)
	sqlite_connection = engine.connect()
	df.to_sql('DisasterResponse', sqlite_connection, if_exists='replace', index=False)
	return ("File Saved!")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()