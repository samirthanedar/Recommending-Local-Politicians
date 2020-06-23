# Recommending local politicians to vote for based on your Twitter profile

## Introduction

I've always been someone who follows politics at a presidential level but not a local level. I vote in general elections but am less likely to vote in midterm elections or primaries. However, local politicians are more likely to have an effect on your day-to-day life than the President. President Barack Obama said it best in his recent Medium post discussing the George Floyd killing. 

> ...the elected officials who matter most in reforming police departments and the criminal justice system work at the state and local levels

However, voter turnout for these elections are extremely low and the winner is usually decided by just a few thousand votes. Thus, in order to help myself better understand which local politicians I agree with, I decided to build this recommendation engine.


## Objective

The goal is to build a recommendation engine for politicians running for office in the Bay Area. The system takes a twitter username and the Bay Area county as input and then returns a recommendation for who to vote for each political contest in that county.  


## Methodology

I used GetOldTweets3 API to scrape tweets from 66 politicians. For each politician, I used TF-IDF to create a document term matrix and then I summed up the values in each term column to end up with a single row vector for each politican (i.e. a politician-term matrix). I used VaderSentiment to add sentiment values for tweets containing the top 200 words. Finally, I used cosine similarity to find the most similar politician to the input twitter handle to make the final recommendation.

## Findings:

The model was able to separate democrats and republicans well and thus for moderate candidates it produces adequate recommendations. Sentiment analysis helps make better recommendations in most cases. However, the model with sentiment sometimes struggles to place far right and far left politicians. Occassionaly even, a far left person will get a recommendation to vote for a far right person. I think perhaps far left and far right people are both expressing a similar sentiment: distaste of the status quo government and for that reason the model thinks they are similar. Finally, the model does tend to be biased towards recommending democrats because there are more democratic politicians in the Bay Area and the democrats here tend to have more tweets than the republicans.


## Navigating the Project Files:

You should follow the following workflow when going through my work to repeat the results: 
1. Run ```python3 twitter_recommender_app.py ``` in command line. You may need to install Flask and a few other packages first. This file contains the main function that outputs the recommendation. It is called "whole_enchilada"
1. Then go to the port that it states to see the web app in your browser. You should be able to enter in a twitter handle and a county and get results back.
1. Additionally, the "whole_enchilada" function refers to many other helper functions I wrote which you'll find in twitter_recommender_api.py

I've also included two jupyter notebooks showing how I built the NLP pipeline and how I tested my results. You should start with project_5_politician_groups-w_sentiment_analysis.ipynb and then move to project_5_recommendation_engine. 

Finally, I have also included pickle files that has some of the data saved so the model runs faster.

