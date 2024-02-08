import pandas as pd
from itertools import product

MLM_MODEL = "mlm3"
FEATURES = ["cosine_tag_review_mean", "cosine_tag_review_max", "cosine_title", "cosine_reviews", "cosine_tag"]
COMBINATIONS = list(product([True, False], repeat=len(FEATURES)))

def select_features():
    selected = []
    for feature in feature_set.keys():
        if feature_set[feature]:
            selected.append(feature)
    return selected 

results = []
for i, combination in enumerate(COMBINATIONS):
    print(f"combination: {i}")
    feature_set = dict(zip(FEATURES, combination))
    selected_features = select_features()
    prediction_folder = "_".join(["tag_dl"] + selected_features)
    maes = []

    for TENFOLD in range(10):
        test = pd.read_csv(f"movie_dataset_public_final/processed/10folds/test{TENFOLD}.csv").rename(columns={"movieId" : "item_id"})
        test = test[["tag", "targets", "item_id"]] 

        predictions = pd.read_csv(f"predictions/{prediction_folder}/predictions{TENFOLD}.csv")
        predictions.predictions = predictions.predictions * 4 + 1

        evaluation = pd.merge(test, predictions, on=["tag", "item_id"], how="left")
        evaluation["error"] = evaluation.targets - evaluation.predictions
        mae = evaluation['error'].abs().sum() / len(evaluation)
        maes.append(mae) 

    results.append({**feature_set, 'overall_mae': sum(maes)/len(maes)})

result_df = pd.DataFrame(results)
result_df = result_df.sort_values(by="overall_mae")

result_df.to_csv('result_summary.csv', index=False)