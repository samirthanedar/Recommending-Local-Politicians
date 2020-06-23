import pandas as pd
import numpy as np
import re
import string
import pickle
import flask
from flask import request

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from twitter_recommender_api import get_politician_vectors, get_wordnet_pos, text_clean, create_sentiment_vectors, create_similarity_matrix, get_similarities, compile_politician_df,scrape_tweets, recommendation 
import GetOldTweets3 as got

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

app = flask.Flask(__name__)

with open("all_politicians_in_california.pkl", "rb") as f:
    all_elections_ca = pickle.load(f)

with open("all_counties_vectorized.pkl", "rb") as f:
    all_counties_vectorized = pickle.load(f)

with open("main_politicians_vectorized.pkl", "rb") as f:
    main_politicians_vectorized = pickle.load(f)

# An example of routing:
# If they go to the page "/" (this means a GET request
# to the page http://127.0.0.1:5000/), return a simple
# page that says the site is up!

@app.route("/", methods=["POST", "GET"])
def whole_enchilada(user_input=True, top_words=True, top_words_amount=200):
    
    #getting the input from the web page
    if request.args.to_dict():
        feature_dict = request.args.to_dict()
        x_input = [(feature_dict.get(name, 0)) for name in ['personal_handle','county_name']]
        personal_handle=x_input[0]
        county_name = x_input[1]

    #if there are no args then we use some generic input
    else:
        personal_handle = 'JoeBiden'
        county_name = 'San Francisco'

    #retrieving the dataframe and number of tweets dictionary for the specific county from the pickle file
    for county in all_counties_vectorized:
        if county[0] == county_name:
        	df = county[1]
        	number_tweets_dict = county[2]
    
    #compile the full dataframe with word vectors for each politician
    full_politician_df,number_tweets_dict = compile_politician_df(df, personal_handle,
    main_politicians_vectorized = main_politicians_vectorized, number_tweets_dict = number_tweets_dict,user_input=user_input,
     top_words=top_words,top_words_amount=top_words_amount)
    
    #getting cosine similarity metrics for each politician and the twitter handle that the user inputted
    sims = get_similarities(personal_handle, full_politician_df, cosine_similarity)
    
    #adding cosine similarity and number of tweets to the table with all politicians in county
    all_politicians_in_county = all_elections_ca[all_elections_ca.COUNTY_NAME == county_name]
    complete_table = all_politicians_in_county.merge(sims[1:], how='outer',left_on='twitter_handle', right_index=True )
    complete_table['number_of_tweets'] = complete_table['twitter_handle'].map(number_tweets_dict)
    
    #cleaning up the data so it looks nicer when shown
    complete_table['number_of_tweets'] = complete_table['number_of_tweets'].fillna('Limited or No Twitter')

    #getting the recommendation for each political contest in the county
    dataframes,candidates_to_vote_for = recommendation(complete_table=complete_table, personal_handle=personal_handle,num_tweets=number_tweets_dict)

    #plotting each politician's twitter profile in 2 dimensions 
    #reduced_data = PCA(n_components=2).fit_transform(full_politician_df)
    #results = pd.DataFrame(reduced_data,columns=['pca1','pca2'],index=full_politician_df.index)
    
    #plt.figure(figsize=(12,8))
    #sns.scatterplot(x="pca1", y="pca2", s=200,hue=results.index, data=results)
    #plt.title('Political Spectrum')
    #plt.show()
    
    return flask.render_template('twitter_recommender.html', dataframes=dataframes,
    	personal_handle=personal_handle,county_name=county_name, candidates_to_vote_for=candidates_to_vote_for)




if __name__=="__main__":
    # For local development:
    app.run(debug=True)
    # For public web serving:
    #app.run(host='0.0.0.0')
    #app.run()