import numpy as np
import pandas as pd
import torch
from torch import nn
from json import loads
from torch import cuda
import warnings
import os
warnings.filterwarnings("ignore")
from itertools import product

device = "cuda" if cuda.is_available() else "cpu"

tag_path = ""

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-04
ORIGINAL_VECTOR_LENGTH = -1 # the vector length of one hot + original features  
MLM_MODEL = ""
FEATURES = ["cosine_tag_review_mean", "cosine_tag_review_max", "cosine_title", "cosine_reviews", "cosine_tag"]
COMBINATIONS = list(product([True, False], repeat=len(FEATURES)))
OVERWRITE = False



class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout_prob=0.2, activation=nn.ReLU()):
        super(FFNN, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_prob)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        return self.sigmoid(x)
    
    
def train(data, epoch):
    model.train()
    data = data.sample(frac=1, random_state=42)  
    batches_total = int(data.shape[0]/BATCH_SIZE + 1)
    epoch_loss = 0
    for i in range(batches_total):
        batch = data.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        input = torch.cat((torch.tensor(batch[["log_IMDB", "log_IMDB_nostem", "rating_similarity", "avg_rating", "tag_exists", "lsi_tags_75", "lsi_imdb_175", "tag_prob"] + selected_features].values.tolist()), torch.tensor(batch["one_hot"].values.tolist())), dim=1).to(device, dtype=torch.float)
        target = ((torch.tensor(batch["targets"].tolist()) - 1) / 4).to(device, dtype=torch.float)           
        optimizer.zero_grad()
        predicted = model(input).to(device, dtype=torch.float).squeeze()
        loss = loss_fn(predicted, target)
        epoch_loss += loss
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch} train loss: {epoch_loss / batches_total}")   

# take some sample of the training data to run validation, if necessary
def validation(data, epoch):
    model.eval()
    data = data.sample(frac=1, random_state=42)  
    batches_total = int(data.shape[0]/BATCH_SIZE + 1)
    epoch_loss = 0
    for i in range(batches_total):
        batch = data.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        input = torch.cat((torch.tensor(batch[["log_IMDB", "log_IMDB_nostem", "rating_similarity", "avg_rating", "tag_exists", "lsi_tags_75", "lsi_imdb_175", "tag_prob"] + selected_features].values.tolist()), torch.tensor(batch["one_hot"].values.tolist())), dim=1).to(device, dtype=torch.float)
        target = ((torch.tensor(batch["targets"].tolist()) - 1) / 4).to(device, dtype=torch.float)           
        predicted = model(input).to(device, dtype=torch.float).squeeze()
        loss = loss_fn(predicted, target)
        epoch_loss += loss
    print(f"epoch {epoch} validation loss: {epoch_loss / batches_total}")
    return epoch_loss     

def predict(data):
    model.eval()
    with torch.no_grad():
        output_df = pd.DataFrame(columns=["item_id", "tag", "predictions"])
        for i in range(int(data.shape[0]/BATCH_SIZE + 1)):
            batch = data.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            input = torch.cat((torch.tensor(batch[["log_IMDB", "log_IMDB_nostem", "rating_similarity", "avg_rating", "tag_exists", "lsi_tags_75", "lsi_imdb_175", "tag_prob"] + selected_features].values.tolist()), torch.tensor(batch["one_hot"].values.tolist())), dim=1).to(device, dtype=torch.float)        
            predicted = model(input).to(device, dtype=torch.float).squeeze().cpu().tolist()
            batch_df = pd.DataFrame({"item_id": batch["item_id"].tolist(), "tag": batch["tag"].tolist(), "predictions": predicted})
            output_df = pd.concat([output_df, batch_df])
        output_df.to_csv(f"predictions/{prediction_folder}_{MLM_MODEL}/predictions{TENFOLD}.csv", index=False)

def get_tag_indexing():
    with open(tag_path) as file:
        tag_dict = {}
        tag_index = []
        for i, line in enumerate(file):
            json_dict = loads(line)
            tag_dict[json_dict["tag"]] = i
            tag_index.append(json_dict["id"])      
        return tag_dict, tag_index                  

def get_one_hot(tag):
    vector = len(tag_index) * [0]
    vector[tag_dict[tag]] = 1
    return vector

def scale(x):
    return x * (SCALING_MAX - SCALING_MIN) + SCALING_MIN

def select_features():
    selected = []
    for feature in feature_set.keys():
        if feature_set[feature]:
            selected.append(feature)
    return selected 

def create_prediction_folder():
    if not os.path.exists(f"predictions/{prediction_folder}_{MLM_MODEL}"):
        os.makedirs(f"predictions/{prediction_folder}_{MLM_MODEL}")


for i, combination in enumerate(COMBINATIONS):

    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"combination: {i}")
    feature_set = dict(zip(FEATURES, combination))
    selected_features = select_features()
    print(f"selected features: {','.join(selected_features)}")
    prediction_folder = "_".join(["tag_dl"] + selected_features)
    if os.path.exists(f"predictions/{prediction_folder}_{MLM_MODEL}") and OVERWRITE == False:
        print(f"{combination} already run, skipping")
        continue
    create_prediction_folder()
    for TENFOLD in range(10):
        print(f"tenfold: {TENFOLD}")
        tag_dict, tag_index = get_tag_indexing()
        SCALING_MIN = -0.7506188543104041 #negative value for tag exists
        SCALING_MAX = 1.33220801542758 #positive value for tag exists
        SCALING_MIN_BOOKS = -0.5462734591669071
        SCALING_MAX_BOOKS = 1.83056878346075
        train_data = pd.read_csv(f"train{TENFOLD}_combined_{MLM_MODEL}.csv").rename(columns={"movieId": "item_id"})
        test_data = pd.read_csv(f"test{TENFOLD}_combined_{MLM_MODEL}.csv").rename(columns={"movieId": "item_id"})
        train_data["one_hot"] = train_data["tag"].apply(lambda t: get_one_hot(t))
        test_data["one_hot"] = test_data["tag"].apply(lambda t: get_one_hot(t))
        train_data[selected_features] = train_data[selected_features].apply(scale, axis=0)
        test_data[selected_features] = test_data[selected_features].apply(scale, axis=0)
        print(train_data.shape)
        print(test_data.shape)
        model = FFNN({ORIGINAL_VECTOR_LENGTH} + len(selected_features), hidden_size = 64)
        model.to(device)
        loss_fn = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

        for epoch in range(EPOCHS):
            train(train_data, epoch)
            validation(test_data, epoch)

        predict(test_data)

