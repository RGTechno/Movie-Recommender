import pandas as pd 

columns = ["user_id","movie_id","rating","timestamp"]
df = pd.read_csv("./ml-100k/u.data",sep="\t",names=columns)

movie_titles = pd.read_csv("./ml-100k/u.item",sep="\|",engine="python",header=None)

movie_titles = movie_titles[[0,1]]
movie_titles.columns = ["movie_id","title"]

df = pd.merge(df,movie_titles,on="movie_id")

ratings = pd.DataFrame(df.groupby("title").mean()["rating"])
ratings["num of ratings"] = pd.DataFrame(df.groupby("title").count()["rating"])

movieMatrix = df.pivot_table(index="user_id",columns="title",values="rating")

def recommedMovie(movieName):
    movie_user_ratings = movieMatrix[movieName]
    movie_corr = movieMatrix.corrwith(movie_user_ratings)
    similar_to_movie = movie_corr
    corr_movie = pd.DataFrame(similar_to_movie,columns=["Correlation"])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings["num of ratings"]).sort_values("Correlation",     ascending=False)                  
    prediction = corr_movie[corr_movie["num of ratings"]>100]

    return prediction

predictions = recommedMovie("Titanic (1997)")

print(predictions.head())