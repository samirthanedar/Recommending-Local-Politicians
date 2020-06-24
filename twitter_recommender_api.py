import pandas as pd
import numpy as np
import re
import pickle
import string
import GetOldTweets3 as got

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import jaccard_score


import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.corpus import wordnet

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

#lemmatization function taken from Selva Prabhakaran's post on Machine Learning Plus
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def text_clean(data):
    """
    Purpose: Takes tweets in a Series, cleans each tweet and returns an array with cleaned tweets.
    Arguments: dataframe of tweets for a column called "text" containing the text of the tweet
    Returns: cleaned array of tweets
    """
    df = data.drop_duplicates()
    
    #taking our URLs
    urls = lambda x: re.sub(r'http\S+', '' ,str(x))

    #taking out capitalization and digits
    alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', str(x))

    #removing punctuation
    punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', str(x).lower())
    
    #applies the above three functions on the data
    df = df.map(urls).map(alphanumeric).map(punc_lower)
    
    #turn data into a list of tweets
    data_list = [x for x in df]
    
    #initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    #lemmatize the words
    data = []
    for sentence in data_list:
        data.append([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])
    
    #combine the words back into tweets
    final = []
    for sentence in data:
        final.append(' '.join(sentence))
    
    return final

def create_sentiment_vectors(list_of_tweets,handle,word=None):
    """
    Purpose: Get total positive, neutral, negative and compound sentiment scores for each tweet and then return the sum
    of all those scores in a single vector for each politician
    Arguments: List of tweets
    Returns: One row dataframe with total sentiment scores for that politician
    """
    
    #Initializing variables
    neg = 0
    neu = 0
    pos = 0 
    compound = 0
    
    #initializing sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    #performing sentiment analysis on a tweet by tweet basis
    for val in list_of_tweets:
        sentiment_dict = analyzer.polarity_scores(val)
        neg += sentiment_dict['neg']
        neu += sentiment_dict['neu']
        pos += sentiment_dict['pos']
        compound += sentiment_dict['compound']
    
    #conditional logic that will change behavior based on if this sentiment vector is only for one word or the whole df
    if word is not None:
        final = pd.DataFrame([[neg,neu,pos,compound]], index=[handle],
                             columns= [word + '_' + val for val in sentiment_dict.keys()])
    else:
        final = pd.DataFrame([[neg,neu,pos,compound]], index=[handle], columns= sentiment_dict.keys())
        
    return final

def get_politician_vectors(twitter_handle, user_input=False,top_words=False,top_words_amount=200):
    """
    Purpose: Take politician tweets from last year and turn it vector of words 
    Arguments: Twitter handle in string, user_input boolean which is true if user input is being accepted, 
    top_words boolean which is true if sentiment analysis is being done on tweets with top words, top_words_amount
    Returns: Dataframe with vector
    """

    #adding custom stop words for this use case
    addl_stop_words = ['live', 'today', 'must', 'join', 'campaign', 'reporter', 'tune', 'pm', 'et', 'press','weekly',
                  'year','thank', 'thanks', 'support', 'appreciate', 'rsvp', 'say', 'get', 'amp']

    custom_stop_words = stopwords.words('english') + addl_stop_words
    
    #load in data
    if user_input == False:
        file = 'data/' + twitter_handle + '_03_2019_to_03_2020.csv'
        df = pd.read_csv(file)
        df = df.text
    else:
        df = scrape_tweets(twitter_handle)
    
    #clean the data
    result = text_clean(df)
    number_of_tweets = len(result)
        
    #create the document-term matrix
    tfidf = TfidfVectorizer(stop_words=custom_stop_words,ngram_range=(1,3), min_df = 5, max_df=.9, binary=True)
    doc_word = tfidf.fit_transform(result)
    doc_word_df = pd.DataFrame(doc_word.toarray(),index=df.drop_duplicates(),columns=tfidf.get_feature_names())
    
    #sum up all columns
    data = np.zeros((1,len(doc_word_df.columns)))
    for i,column in enumerate(doc_word_df.columns):
        data[0,i] = doc_word_df[column].sum()
    
    #put the sums into a new dataframe
    final = pd.DataFrame(data,index=[twitter_handle], columns=doc_word_df.columns)
        
    #get sentiment for top words 
    if top_words == True:
        top_words = list(final.T.sort_values(by=twitter_handle,ascending=False).index[0:top_words_amount])
        for word in top_words:
            mini_df = pd.DataFrame(result,columns=['text'])
            tweets_for_this_word = mini_df[mini_df.text.str.contains(word)].text.to_list()
            if len(tweets_for_this_word) != 0:
                senti_df = create_sentiment_vectors(tweets_for_this_word,twitter_handle,word)
                for column in senti_df.columns:
                    final[column] = senti_df.at[twitter_handle,column]

    return final,number_of_tweets

def compile_politician_df(politician_df, personal_handle, main_politicians_vectorized, number_tweets_dict,user_input=False, top_words=False,top_words_amount=200):
    """
    Purpose: Add the twitter username inputted by the user into the full politician dataframe
    Arguments: the existing dataframe with vectors for each politician in the county, the personal twitter username to 
    add to the list, some saved vectors for popular politicians that I was testing with, 
    the exisiting number of tweets dictionary, user_input boolean determining if we should scrape 
    twitter or not, and then two more variables that tells the function whether or not to include sentiment or not.
    Returns: the full dataframe and a dictionary showing how many tweets each politician had
    """


    #get the vector for the twitter username entered by the user
    if personal_handle in ['JoeBiden', 'realDonaldTrump', 'AOC']:
        for politician in main_politicians_vectorized:
            if politician[0] == personal_handle:
                new_df = politician[1]
                number_of_tweets = politician[2]
    else:
        new_df,number_of_tweets = get_politician_vectors(personal_handle,user_input=True, top_words=top_words,top_words_amount=top_words_amount)
   
    #add the number of tweets to the exisiting dictionary
    number_tweets_dict[personal_handle] = number_of_tweets

    #adding the vector to the big dataframe
    big_df = pd.concat([politician_df, new_df], axis=0)

    #fill NA values    
    full_politician_df = big_df.fillna(0)

    return full_politician_df,number_tweets_dict

def create_similarity_matrix(similarity_tool,df_vectors,personal_handle):
    """
    Purpose: Take a dataframe with vectors of politicians and output the similarities between the personal handle and everyone else
    Arguments: similarity_tool: cosine similarity or euclidean distance, a dataframe with vectors, personal handle
    Returns: a brand new dataframe with similarity scores
    """
    
    similarities = []
    first_vec = df_vectors.loc[personal_handle,:].values.reshape(1,-1)
    iters = len(df_vectors.index)
    for j in range(iters):
        second_vec = df_vectors.iloc[j,:].values.reshape(1,-1)
        similarities.append(similarity_tool(first_vec,second_vec)[0][0])


                
    df = pd.DataFrame(similarities, index=df_vectors.index, columns = [personal_handle])      

    return df

def scrape_tweets(twitter_handle):
    """
    Purpose: Scrape tweets from 03/2019 to 03/2020 for an inputted twitter handle
    Arguments: twitter handle in a string
    Returns: a pandas series of tweets
    """

    #using the GetOldTweets API
    tweetCriteria = got.manager.TweetCriteria().setUsername(twitter_handle)\
                                           .setSince("2019-03-03")\
                                           .setUntil("2020-03-03").setMaxTweets(1000)
    
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    
    tweet_list = []
    for tweet in tweets:
        tweet_list.append(tweet.text)
        
    result = pd.Series(tweet_list).drop_duplicates()
    return result

def get_similarities(twitter_handle,politician_df,similarity_tool):
    """
    Purpose: sorts similarties based on which similarity tool is being used
    Arguments: personal twitter handle, a dataframe with vectors, and similarity_tool: cosine similarity or euclidean distance
    Returns: a sorted dataframe with the most similar politicians being at the top
    """

    sim_df = create_similarity_matrix(similarity_tool,politician_df,twitter_handle)
    if similarity_tool == cosine_similarity:
        return sim_df[twitter_handle].sort_values(ascending=False)
    if similarity_tool == euclidean_distances:
        return sim_df[twitter_handle].sort_values()

def recommendation(complete_table, personal_handle, num_tweets):
    """
    Purpose: Based on similarity metric, get a recommendation for who the user should vote for in each race in the county.
    Also, return a sorted list of all politicians starting with most similar to least similar
    Arguments: complete_table = a dataframe of all politicians in the county with similarity metric,
    twitter_handles and number of tweets already added, personal_handle = the twitter usename for the profile
    we are comparing to politicians
    Returns: a list of dataframes showing the full results for each contest and a list of lists 
    corresponding to a single recommendation for each contest 

    """
    dataframes = []
    candidates_to_vote_for = []

    twitter_only = complete_table[complete_table.twitter_handle != 'Limited or No Twitter']

    for contest in complete_table['CONTEST_NAME'].unique():
        if contest not in twitter_only.CONTEST_NAME.unique():
            print(contest)
            print('No politicians in this race have a Twitter account!\n')
        else:
            #set up dataframe for displaying
            full_df = complete_table[complete_table.CONTEST_NAME==contest].sort_values(by=personal_handle,ascending=False).reset_index(drop=True)
            full_df[personal_handle] = round(full_df[personal_handle],2)
            full_df[personal_handle] = full_df[personal_handle].fillna('Limited or No Twitter')
            
            #grab information for top recommendation
            reco_df = twitter_only[twitter_only.CONTEST_NAME==contest].sort_values(by=personal_handle,ascending=False).reset_index(drop=True)
            df = full_df[['CANDIDATE_NAME', 'twitter_handle', 'PARTY_NAME', 'number_of_tweets', personal_handle ]]
            df = df.rename(columns={"CANDIDATE_NAME": "Candidate", 'twitter_handle': "Twitter Handle",
                'PARTY_NAME': 'Party', 'number_of_tweets': 'Number of tweets', personal_handle:'Similarity' })
            candidate_contest = str(contest) + ': ' + str(reco_df.at[0,'CANDIDATE_NAME'])
            candidates_to_vote_for.append(candidate_contest)
            dataframes.append([contest,df.to_html()])

            #for debugging purposes
            print(contest)
            print(full_df[['COUNTY_NAME','CANDIDATE_NAME', 'PARTY_NAME', personal_handle, 'number_of_tweets']])
            print('\n')

    return dataframes,candidates_to_vote_for