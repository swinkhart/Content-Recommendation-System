import pandas as pd

contentDf = pd.read_csv("datasets/Anime.csv")

# the number of episodes is an issue with the dataset as some are incomplete and are unimportant for the reccomendation so they're just dropped, the anime_id is also umimportant so it is also dropped
contentDf.drop(columns=["Japanese_name", "Episodes", "Studio", "Release_season", "Release_year", "End_year", "Description", "Content_Warning", "Related_Mange", "Related_anime", "Voice_actors", "staff"], inplace=True)

# convert string numbers to actual, operable numbers
contentDf["Rating"] = pd.to_numeric(contentDf["Rating"])
contentDf["Rank"] = pd.to_numeric(contentDf["Rank"])

# dropping content with unknown ratings and tags, filling with an average rating could lead to inaccurate recommendations and it is not possible to guess tags
contentDf.dropna(axis=0, how="all", subset=["Tags"], inplace=True)
contentDf.dropna(axis=0, how="all", subset=["Rating"], inplace=True)

#dropping odd or uncatigorized types
contentDf.drop(contentDf.loc[contentDf["Type"] == "TV Sp"].index, inplace=True)
contentDf.drop(contentDf.loc[contentDf["Type"] == "DVD S"].index, inplace=True)
contentDf.drop(contentDf.loc[contentDf["Type"] == "Other"].index, inplace=True)

# creating dataframe to use to create the fit for the
fitDataset = contentDf.drop(columns=["Name"])

# concatinating the genre and type as dummy vatiables rather than cataigories so that they can be used for calculation
fitDataset = pd.concat(
    [fitDataset.drop("Tags", axis=1), fitDataset["Tags"].str.get_dummies(sep=", ")],
    axis=1,
)
fitDataset = pd.concat(
    [fitDataset.drop("Type", axis=1), pd.get_dummies(fitDataset["Type"])], axis=1
)

# create new cleaned dataset for use in the recommendation system
contentDf.to_csv("datasets/cleanedAnime.csv", sep=",")
fitDataset.to_csv("datasets/fitSet.csv", sep=",")