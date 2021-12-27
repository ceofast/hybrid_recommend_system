# HYBRID RECOMMENDER SYSTEM

# Bussines Problem
# Make an estimate for the user whose ID is given,
# using the item-based and user-based recommender methods.

# Dataset Story

# The dataset was provided by MovieLens, a movie recommendation service.
# It contains the rating scores for these movies along with the movies.
# It contains 2,000,0263 ratings across 27,278 movies.
# This data was created by 138,493 users between 09 January 1995 and
# 31 March 2015. This data set was created on October 17, 2016.
# Users are randomly selected. It is known that all selected users voted
# for at least 20 movies.

# Variables

# movie.csv

# movield: Unique movie number (UniqueID)
# title: Movie name

# rating.csv

# userId: Unique user number(UniqueID)
# movieId: Unique film number (UniqueID)
# rating: Rating given to the movie by the user
# timestamp: Evaluation date

# Task 1: Perform data preparation operations.

import pandas as pd
pd.set_option("display.max_columns", 20)
movie = pd.read_csv("/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_4/movie_lens_dataset/movie.csv")
movie.shape
# (27278, 3)
movie.head()
#    movieId                               title  \
# 0        1                    Toy Story (1995)   
# 1        2                      Jumanji (1995)   
# 2        3             Grumpier Old Men (1995)   
# 3        4            Waiting to Exhale (1995)   
# 4        5  Father of the Bride Part II (1995)   
#                                        genres  
# 0  Adventure|Animation|Children|Comedy|Fantasy  
# 1                   Adventure|Children|Fantasy  
# 2                               Comedy|Romance  
# 3                         Comedy|Drama|Romance  
# 4                                       Comedy  

rating = pd.read_csv("/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_4/movie_lens_dataset/rating.csv")
rating.shape
# (20000263, 4)
rating.head()
#    userId  movieId  rating            timestamp
# 0       1        2     3.5  2005-04-02 23:53:47
# 1       1       29     3.5  2005-04-02 23:31:16
# 2       1       32     3.5  2005-04-02 23:33:39
# 3       1       47     3.5  2005-04-02 23:32:07
# 4       1       50     3.5  2005-04-02 23:29:40

df = movie.merge(rating, how="left", on="movieId")
df.head()
#     movieId             title                                       genres  \
# 0        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy   
# 1        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy   
# 2        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy   
# 3        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy   
# 4        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy   
#    userId  rating            timestamp  
# 0     3.0     4.0  1999-12-11 13:36:47  
# 1     6.0     5.0  1997-03-13 17:50:52  
# 2     8.0     4.0  1996-06-05 13:37:51  
# 3    10.0     4.0  1999-11-25 02:44:47  
# 4    11.0     4.5  2009-01-02 01:13:41  

df.shape
# (20000797, 6)
df["title"].nunique()
# 27262
df["title"].value_counts().head()
# Pulp Fiction (1994)                 67310
# Forrest Gump (1994)                 66172
# Shawshank Redemption, The (1994)    63366
# Silence of the Lambs, The (1991)    63299
# Jurassic Park (1993)                59715

comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
comment_movies = df[~df["title"].isin(rare_movies)]
comment_movies.shape
# (17766015, 6)
comment_movies["title"].nunique()
# 3159
user_movie_df = comment_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.shape
# (138493, 3159)
user_movie_df.head()
#  title   'burbs, The (1989)  (500) Days of Summer (2009)  \
# userId                                                    
# 1.0                    NaN                          NaN   
# 2.0                    NaN                          NaN   
# 3.0                    NaN                          NaN   
# 4.0                    NaN                          NaN   
# 5.0                    NaN                          NaN   
# title   *batteries not included (1987)  ...And Justice for All (1979)  \
# userId                                                                  
# 1.0                                NaN                            NaN   
# 2.0                                NaN                            NaN   
# 3.0                                NaN                            NaN   
# 4.0                                NaN                            NaN   
# 5.0                                NaN                            NaN   
# title   10 Things I Hate About You (1999)  10,000 BC (2008)  \
# userId                                                        
# 1.0                                   NaN               NaN   
# 2.0                                   NaN               NaN   
# 3.0                                   NaN               NaN   
# 4.0                                   NaN               NaN   
# 5.0                                   NaN               NaN   
# title   101 Dalmatians (1996)  \
# userId                          
# 1.0                       NaN   
# 2.0                       NaN   
# 3.0                       NaN   
# 4.0                       NaN   
# 5.0                       NaN   
# title   101 Dalmatians (One Hundred and One Dalmatians) (1961)  \
# userId                                                           
# 1.0                                                   NaN        
# 2.0                                                   NaN        
# 3.0                                                   NaN        
# 4.0                                                   NaN        
# 5.0                                                   NaN        
# title   102 Dalmatians (2000)  12 Angry Men (1957)  ...  \
# userId                                              ...   
# 1.0                       NaN                  NaN  ...   
# 2.0                       NaN                  NaN  ...   
# 3.0                       NaN                  NaN  ...   
# 4.0                       NaN                  NaN  ...   
# 5.0                       NaN                  NaN  ...   
# title   Zero Dark Thirty (2012)  Zero Effect (1998)  Zodiac (2007)  \
# userId                                                               
# 1.0                         NaN                 NaN            NaN   
# 2.0                         NaN                 NaN            NaN   
# 3.0                         NaN                 NaN            NaN   
# 4.0                         NaN                 NaN            NaN   
# 5.0                         NaN                 NaN            NaN   
# title   Zombieland (2009)  Zoolander (2001)  Zulu (1964)  [REC] (2007)  \
# userId                                                                   
# 1.0                   NaN               NaN          NaN           NaN   
# 2.0                   NaN               NaN          NaN           NaN   
# 3.0                   NaN               NaN          NaN           NaN   
# 4.0                   NaN               NaN          NaN           NaN   
# 5.0                   NaN               NaN          NaN           NaN   
# title   eXistenZ (1999)  xXx (2002)  ¡Three Amigos! (1986)  
# userId                                                      
# 1.0                 NaN         NaN                    NaN  
# 2.0                 NaN         NaN                    NaN  
# 3.0                 NaN         NaN                    NaN  
# 4.0                 NaN         NaN                    NaN  
# 5.0                 NaN         NaN                    NaN  

user_movie_df.columns
# Index([''burbs, The (1989)', '(500) Days of Summer (2009)',
#       '*batteries not included (1987)', '...And Justice for All (1979)',
#       '10 Things I Hate About You (1999)', '10,000 BC (2008)',
#       '101 Dalmatians (1996)',
#       '101 Dalmatians (One Hundred and One Dalmatians) (1961)',
#       '102 Dalmatians (2000)', '12 Angry Men (1957)',
#       ...
#       'Zero Dark Thirty (2012)', 'Zero Effect (1998)', 'Zodiac (2007)',
#       'Zombieland (2009)', 'Zoolander (2001)', 'Zulu (1964)', '[REC] (2007)',
#       'eXistenZ (1999)', 'xXx (2002)', '¡Three Amigos! (1986)'],

len(user_movie_df.columns)
# 3159
comment_movies["title"].nunique()
# 3159

# Task 2: Determine the movies watched by the user to be suggested.

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)
random_user_df = user_movie_df[user_movie_df.index == random_user]
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
len(movies_watched)
# 33
user_movie_df.loc[user_movie_df.index == random_user, user_movie_df.columns == "Jurassic Park (1993)"]
# title    Jurassic Park (1993)
# userId                       
# 28941.0                   3.0


# Task 3: Access data and Ids of other users watching the same movies.

pd.set_option("display.max_columns", 5)
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
# title   Ace Ventura: Pet Detective (1994)  \
# userId                                      
# 1.0                                   NaN   
# 2.0                                   NaN   
# 3.0                                   NaN   
# 4.0                                   NaN   
# 5.0                                   NaN   
# title   Ace Ventura: When Nature Calls (1995)  ...  \
# userId                                         ...   
# 1.0                                       NaN  ...   
# 2.0                                       NaN  ...   
# 3.0                                       NaN  ...   
# 4.0                                       3.0  ...   
# 5.0                                       NaN  ...   
# title   Star Trek: Generations (1994)  Stargate (1994)  
# userId                                                  
# 1.0                               NaN              NaN  
# 2.0                               NaN              NaN  
# 3.0                               5.0              5.0  
# 4.0                               3.0              NaN  
# 5.0                               NaN              4.0  

movies_watched_df.shape
# (138493, 33)
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)
#           userId  movie_count
# 94230    94231.0           33
# 100398  100399.0           33
# 118204  118205.0           33
# 15918    15919.0           33
# 124051  124052.0           33
#           ...          ...
# 79214    79215.0           21
# 79174    79175.0           21
# 9105      9106.0           21
# 78515    78516.0           21
# 129        130.0           21
user_movie_count[user_movie_count["movie_count"] == len(movies_watched)].count()
# userId         17
# movie_count    17

# Task 4: Identify the users who are most similar to the user to be suggested.

percentage = len(movies_watched)*60/100
percentage
# 19.8
user_same_movies = user_movie_count[user_movie_count["movie_count"] > percentage]["userId"]
user_same_movies.count()
# 4139
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(user_same_movies.index)], random_user_df[movies_watched]])
final_df.head()
# title   Ace Ventura: Pet Detective (1994)  \
# userId                                      
# 90.0                                  3.5   
# 129.0                                 0.5   
# 155.0                                 NaN   
# 157.0                                 NaN   
# 159.0                                 NaN   
# title   Ace Ventura: When Nature Calls (1995)  ...  \
# userId                                         ...   
# 90.0                                      NaN  ...   
# 129.0                                     NaN  ...   
# 155.0                                     NaN  ...   
# 157.0                                     NaN  ...   
# 159.0                                     NaN  ...   
# title   Star Trek: Generations (1994)  Stargate (1994)  
# userId                                                  
# 90.0                              4.5              3.5  
# 129.0                             NaN              NaN  
# 155.0                             NaN              NaN  
# 157.0                             NaN              NaN  
# 159.0                             NaN              NaN 
final_df.shape
# (4140, 33)
final_df.T.corr()
# userId        90.0          129.0     ...  138482.0  28941.0 
# userId                                ...                    
# 90.0      1.000000e+00  8.779946e-01  ...       NaN -0.279828
# 129.0     8.779946e-01  1.000000e+00  ...       NaN  0.154227
# 155.0              NaN           NaN  ...       NaN       NaN
# 157.0     1.000000e+00  1.000000e+00  ...       NaN       NaN
# 159.0     5.000000e-01  1.000000e+00  ...       NaN  0.530330
#                 ...           ...  ...       ...       ...
# 138278.0           NaN           NaN  ...       NaN       NaN
# 138381.0 -3.140185e-16 -3.972055e-16  ...       NaN  0.088697
# 138414.0 -9.614813e-17  6.002450e-01  ...       NaN -0.227710
# 138482.0           NaN           NaN  ...       NaN       NaN
# 28941.0  -2.798283e-01  1.542273e-01  ...       NaN  1.000000
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df=pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names=["user_id_1", "user_id_2"]
corr_df = corr_df.reset_index()
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by="corr", ascending=False)
top_users.rename(columns={"user_id_2":"userId"}, inplace=True)
rating = pd.read_csv("/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_4/movie_lens_dataset/rating.csv")
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")

# Task 5: Calculate the Weighted Average Recommendation Score and keep the first 5 movies.

top_users_ratings["weighted_rating"] = top_users_ratings["corr"] * top_users_ratings["rating"]
top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
#          weighted_rating
# movieId                 
# 1               3.116352
# 2               1.675794
# 3               1.958276
# 6               2.935694
# 7               2.841561
#                   ...
# 89904           2.296264
# 91529           1.640188
# 92259           3.280376
# 95105           0.656075
# 95510           0.328038
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 4].sort_values("weighted_rating", ascending=False)

# Task 6: Make an item-based suggestion based on the name of the movie that the user has watched with the highest score.
# 5 suggestions user-based
# 5 suggestions item-based

movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"].head()
# 0           Happy Gilmore (1996)
# 1               Labyrinth (1986)
# 2    Boondock Saints, The (2000)
# 3                  Snatch (2000)
# 4                 Frailty (2001)
user = 27000
movie = pd.read_csv("/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_4/movie_lens_dataset/movie.csv")
rating = pd.read_csv("/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_4/movie_lens_dataset/rating.csv")
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]
movie_name = movie[movie["movieId"] == movie_id]["title"].values[0]
movie_name = user_movie_df[movie_name]
movies_from_item_based = user_movie_df.corrwith(movie_name).sort_values(ascending=False)
movies_from_item_based[1:6].index
# Index(['Buddy Holly Story, The (1978)', 'Coal Miner's Daughter (1980)',
#       'Lucas (1986)', 'Antwone Fisher (2002)', 'Steel Magnolias (1989)'],
#      dtype='object', name='title')
