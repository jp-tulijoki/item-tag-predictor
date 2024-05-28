import pandas as pd
from itertools import product
from sklearn.metrics import mean_absolute_error
import scipy.stats as st
import numpy as np

EMBEDDINGS_MODEL = ""
BOOKS = False
TEST_FILE_PATH = ""
FEATURES = ["cosine_tag_review_mean", "cosine_tag_review_max", "cosine_title", "cosine_reviews", "cosine_tag"]
COMBINATIONS = list(product([True, False], repeat=len(FEATURES)))

TAG_DL_MAES = [0.85812585499316, 0.8438323236271251, 0.8482134480062549, 0.8493373729476151, 0.8528967943706021, 0.8404437060203284, 0.8483444096950743, 0.8581118060985146, 0.8411630179827989, 0.8230082095387021]
GLMER_MAES = [0.8908792792857654, 0.8653623152293224, 0.8791149480540217, 0.8832295795902697, 0.8891084501402173, 0.8752010166064554, 0.8779947932403129, 0.8801060245174557, 0.8644941026864903, 0.8550020119472006]
AVERAGE_MAES = [1.4415895666551992, 1.4581681687920618, 1.446981316945963, 1.450661051563591, 1.4593910780384587, 1.4647713081696834, 1.4497732780188501, 1.4516386662594638, 1.4587178212547898, 1.462401478251157]
if BOOKS:
    TAG_DL_MAES = [0.7581442111790238, 0.770311807954646, 0.7479679333864824, 0.7521144476924438, 0.7567552484719638, 0.7532730977057313, 0.757920099211622, 0.736154323175053, 0.7460869950389795, 0.7336312898653436]
    GLMER_MAES = [0.8445663125767168, 0.8649455982733268, 0.8519600929556028, 0.8573305084038367, 0.8571838305297244, 0.8513928816721585, 0.8574781595869927, 0.8385652243727156, 0.8448081222785462, 0.8422652560616454]
    AVERAGE_MAES = [1.3970546154774632, 1.4020351252569903, 1.4032667618022594, 1.406772700756396, 1.3932005318409808, 1.407191647758482, 1.4028388535627982, 1.4090232991266254, 1.4056665003838795, 1.3998767573264659]
TAGDL_MEAN = np.mean(TAG_DL_MAES)
GLMER_MEAN = np.mean(GLMER_MAES)
AVERAGE_MEAN = np.mean(AVERAGE_MAES)

def select_features():
    selected = []
    for feature in feature_set.keys():
        if feature_set[feature]:
            selected.append(feature)
    return selected 

results = []
for i, combination in enumerate(COMBINATIONS):

    print(f"combination: {combination}")
    feature_set = dict(zip(FEATURES, combination))
    selected_features = select_features()
    prediction_folder = "_".join(["tag_dl"] + selected_features)
    maes = []

    for TENFOLD in range(10):
        test = pd.read_csv(f"{TEST_FILE_PATH}/test{TENFOLD}.csv").rename(columns={"movieId" : "item_id"})
        test = test[["tag", "targets", "item_id"]] 
        predictions = pd.read_csv(f"predictions/{prediction_folder}_{EMBEDDINGS_MODEL}/predictions{TENFOLD}.csv")
        predictions.predictions = predictions.predictions * 4 + 1
        evaluation = pd.merge(test, predictions, on=["tag", "item_id"], how="left")
        evaluation["error"] = evaluation.targets - evaluation.predictions
        mae = mean_absolute_error(evaluation.predictions, evaluation.targets)
        assert mae == evaluation['error'].abs().sum() / len(evaluation)
        maes.append(mae) 
    overall_mae = sum(maes)/len(maes) 

    tagdl_diff = (TAGDL_MEAN - overall_mae) / TAGDL_MEAN
    
    alpha = 0.05 / 128
    ci = st.t.interval(1 - alpha, len(maes) - 1, loc=overall_mae, scale=st.sem(maes))
    epsilon = (ci[1]-overall_mae)

    _, p_value = st.ttest_ind(maes, TAG_DL_MAES, alternative='less')

    results.append({**feature_set, "overall_mae": overall_mae, "ci +/-": epsilon, "tagdl_diff": tagdl_diff, "p-value": p_value})

ci = st.t.interval(1 - alpha, len(TAG_DL_MAES) - 1, loc=TAGDL_MEAN, scale=st.sem(TAG_DL_MAES))
epsilon = (ci[1]-TAGDL_MEAN)
print(f"tagdl mean: {round(TAGDL_MEAN, 4)}, +/- {round(epsilon, 4)}")
ci = st.t.interval(1 - alpha, len(GLMER_MAES) - 1, loc=GLMER_MEAN, scale=st.sem(GLMER_MAES))
epsilon = (ci[1]-GLMER_MEAN)
print(f"glmer mean: {round(GLMER_MEAN, 4)}, +/- {round(epsilon, 4)}")
ci = st.t.interval(1 - alpha, len(AVERAGE_MAES) - 1, loc=AVERAGE_MEAN, scale=st.sem(AVERAGE_MAES))
epsilon = (ci[1]-AVERAGE_MEAN)
print(f"average mean: {round(AVERAGE_MEAN, 4)}, +/- {round(epsilon, 4)}")

result_df = pd.DataFrame(results)
result_df = result_df.sort_values(by="overall_mae")

result_df.to_csv(f"result_summary_{EMBEDDINGS_MODEL}.csv", index=False)