import pandas as pd

def getAllData(file_ending):
    train = pd.read_csv(f"train0_combined_{file_ending}.csv")
    test = pd.read_csv(f"test0_combined_{file_ending}.csv")
    return pd.concat((train, test))

# change file endings according to your paths
mlm_movies = getAllData("mlm3")
msmarco_movies = getAllData("msmarco")
mlm_books = getAllData("mlm3_books")
msmarco_books = getAllData("msmarco_books")

stats = {
    "set": ["mlm_movies", "msmarco_movies", "mlm_books", "msmarco_books"],
    "mean(cos(t, R)) mean": [df["cosine_tag_review_mean"].mean() for df in [mlm_movies, msmarco_movies, mlm_books, msmarco_books]],
    "mean(cos(t, R)) max": [df["cosine_tag_review_mean"].max() for df in [mlm_movies, msmarco_movies, mlm_books, msmarco_books]],
    "mean(cos(t, R)) min": [df["cosine_tag_review_mean"].min() for df in [mlm_movies, msmarco_movies, mlm_books, msmarco_books]],
    "mean(cos(t, R)) std": [df["cosine_tag_review_mean"].std() for df in [mlm_movies, msmarco_movies, mlm_books, msmarco_books]],
    "max(cos(t, R)) mean": [df["cosine_tag_review_max"].mean() for df in [mlm_movies, msmarco_movies, mlm_books, msmarco_books]],
    "max(cos(t, R)) max": [df["cosine_tag_review_max"].max() for df in [mlm_movies, msmarco_movies, mlm_books, msmarco_books]],
    "max(cos(t, R)) min": [df["cosine_tag_review_max"].min() for df in [mlm_movies, msmarco_movies, mlm_books, msmarco_books]],
    "max(cos(t, R)) std": [df["cosine_tag_review_max"].std() for df in [mlm_movies, msmarco_movies, mlm_books, msmarco_books]],
}

stats_df = pd.DataFrame(stats)

print(stats_df.transpose().to_latex(multirow=True))