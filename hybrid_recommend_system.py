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
movie.head()

rating = pd.read_csv("/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_4/movie_lens_dataset/rating.csv")
rating.shape
rating.head()

df = movie.merge(rating, how="left", on="movieId")
df.head()
df.shape

df["title"].nunique()
df["title"].value_counts().head()

comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
comment_movies = df[~df["title"].isin(rare_movies)]
comment_movies.shape
comment_movies["title"].nunique()

user_movie_df = comment_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.shape
user_movie_df.head()

user_movie_df.columns
len(user_movie_df.columns)
comment_movies["title"].nunique()

# Task 2: Determine the movies watched by the user to be suggested.

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)
random_user_df = user_movie_df[user_movie_df.index == random_user]
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
len(movies_watched)
user_movie_df.loc[user_movie_df.index == random_user, user_movie_df.columns == "Jurassic Park (1993)"]

# Task 3: Access data and Ids of other users watching the same movies.

pd.set_option("display.max_columns", 5)
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)
user_movie_count[user_movie_count["movie_count"] == len(movies_watched)].count()

# Task 4: Identify the users who are most similar to the user to be suggested.

percentage = len(movies_watched)*60/100
percentage
user_same_movies = user_movie_count[user_movie_count["movie_count"] > percentage]["userId"]
user_same_movies.count()
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(user_same_movies.index)], random_user_df[movies_watched]])
final_df.head()
final_df.shape
final_df.T.corr()
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
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 4].sort_values("weighted_rating", ascending=False)

# Task 6: Make an item-based suggestion based on the name of the movie that the user has watched with the highest score.
# 5 suggestions user-based
# 5 suggestions item-based

movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"].head()
user = 27000
movie = pd.read_csv("/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_4/movie_lens_dataset/movie.csv")
rating = pd.read_csv("/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_4/movie_lens_dataset/rating.csv")
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]
movie_name = movie[movie["movieId"] == movie_id]["title"].values[0]
movie_name = user_movie_df[movie_name]
movies_from_item_based = user_movie_df.corrwith(movie_name).sort_values(ascending=False)
movies_from_item_based[1:6].index