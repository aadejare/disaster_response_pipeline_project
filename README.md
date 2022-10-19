# Disaster Response Pipeline Project

Turns text into most relevant topics

##Summary
This pipeline will clean the data to create a cross category of genre and topics to allow 
for better classification
It will then train a Extreme Gradient Boosting model and output the relevant topics.
You can see the most relevant topics in the web browser pages and even see which topic
is viable.  If no prediction is possible, it will let you know

### Instructions:
1. Install NLTK and run 
	`nltk.download('omw-1.4')`

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/
