"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("vect.pkl","rb")#waiting for team4 vectorizer
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit"""

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information",'EDA']
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.info('About this Utility App')
		st.markdown(
			"This Utility app requires the user to input text\
			ideally a tweet relating to climate change), and will\
			classify it according to whether or not they believe in\
			climate change.Below you will find information about the data source\
			and a brief data description. You can have a look at word clouds and\
			other general EDA on the EDA page, and make your predictions on the\
			prediction page that you can navigate to in the sidebar."
			)
		st.info('Data description as per source')
		st.markdown(
			'The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo.\
			This dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were annotated. Each tweet is labelled independently by 3 reviewers.\
			This dataset only contains tweets that all 3 reviewers agreed on (the rest were discarded).\
			Each tweet is labelled as one of the following classes:\
			- 2(News): the tweet links to factual news about climate change\
			- 1(Pro): the tweet supports the belief of man-made climate change\
			- 0(Neutral): the tweet neither supports nor refutes the belief of man-made climate change\
			- -1(Anti): the tweet does not believe in man-made climate change'
		)

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Tweet your tweet of interest","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("final_lsvc.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))
	if selection =='EDA':
		st.title('Exploratory Data Analysis')
		st.markdown('Extensive background about the use of this utility and the origin of the data set has been given on the infomation\
			       page.Here we will explore relationships in the data and the implication of these in the perfomance of the \
			       the model in classifying the sentiments of tweets on climate change')

		st.info('Distribution of tweet sentiments')
		st.image('sentimentcount.png')
		st.image('sentimentdist.png')
		st.markdown(
			'A density plot is a representation of the distribution of a numeric variable .\
			Looking at the four density graphs(for positive,negative,neutral and news) \
			above we realise that the graphs above are not normally distributed,\
			this means that they are not symetrical about the center and tend to be slightly skewed.\
			Taking a closer look at the negative density plot ,\
			we realise that it has a slightly higher density than the rest of the graphs.\
			Therefore the density count is signifigant for anti-climate change'
			)

		st.info('Message length comparisons Across All Sentiment Classes')
		st.image('msglen.png')
		st.markdown(
		"Sentiment analysis is a technique that detects the underlying sentiment in a piece of text.\
	    It is the process of classifying text as either positive, negative, neutral or news.\
		Observing the bar graph above of average length of message by sentiment:\
		We observe that:\
		* the postive sentiment(pro-climate change ) has a significantly higher length above 120\
		* the negative sentiment (anti-climate change) also has a high length of sligtly above 120\
		* the news sentiment (tweets posted by news) has a length of 120\
		* the neutral sentiment( tweets with neutral massege) has a length of slightly above 100 they don't write long tweets as conmpared to the rest")

		st.info('Anticlimate Sentiments Word Cloud')
		st.image('hanticlimate.png')
		st.markdown(
			"We see a lot of negative words such as `fake`,`scam`,`hoax`, `manipulated`, `manmade`, `chinese`.etc\
			These are words which are mostly used by people who are against climate change.\
			We can also see  Donald Trump since he doesnt believe in climate change and he was vocal about his believes."
			)

		st.info('Proclimate Sentiments Word Cloud')
		st.image('hproclimate.png')
		st.markdown(
			"It looks like the news tweets are mostly retweets, we see words like `climatechange`, `environment`,\
			`change`, `climate`, which is what we would expect considering the project is about climate change\
			but we see words like `research`, `washingtonpost`, `cnn`,\
			`scientist`, `study`, `expert`, `policies` which are unique to the news tweets.")

		st.info('News Headlines Word Cloud')
		st.image('hnews.png')
		st.markdown(
			"Here we see a lot of positive words, words that calls for help or action,\
			 words such as `believe`, `action`, `real`, `think`, `environment`, ... etc.")

		st.info('Nuetral Sentiments Word Cloud')
		st.image('hnuetral.png')
		st.markdown(
			"Here we see words like such as `penguin`, `people`, `global` which are classified as neutral sentiment tweets.\
			and we see words like `retweet` which suggest that also most of the tweets are retweets.\
			the neutral tweets, they normally tweet about anything"
		)

		st.info('Top 20 Climate Change Hashtags')
		st.image('gtop20#.png')
		st.markdown(
			'Here we see words like such as `penguin`, `people`, `global` which are classified as neutral sentiment tweets.\
			and we see words like `retweet` which suggest that also most of the tweets are retweets. the neutral tweets,\
			they normally tweet about anything'
		)
		st.info('Top 20 Climate Change News Headlines ')
		st.image('gtop20n.png')
		st.markdown(
			'It looks like the news tweets are mostly retweets, we see words like `climatechange`, `environment`, `change`,\
			`climate`, which is what we would expect considering the project is about climate change but we see words like `research`,\
			`washingtonpost`, `cnn`,`scientist`, `study`, `expert`, `policies` which are unique to the news tweets.'
		)
		st.info('Top 20 positive climate change tweets')
		st.image('gtop20pl.png')
		st.markdown(
			'see a lot of positive words, words that calls for help or action, words such as `believe`, `action`, `real`, `think`, `environment`, ... etc.'
		)

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
