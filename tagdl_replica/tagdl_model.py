import numpy as np
import pandas as pd
import torch
from torch import nn
from json import loads
from torch import cuda
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if cuda.is_available() else 'cpu'

tag_path = 'tags.json'

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-04
TENFOLD = 0

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size = 64, activation=nn.ReLU()):
        super(FFNN, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.activation_1 = activation
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.activation_2 = activation
        self.linear_3 = nn.Linear(hidden_size, hidden_size)
        self.activation_3 = activation
        self.output = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation_1(x)
        x = self.linear_2(x)
        x = self.activation_2(x)
        x = self.linear_3(x)
        x = self.activation_3(x)
        x = self.output(x)
        return self.sigmoid(x)
    
def train(data, epoch):
    data = data.sample(frac=1)  
    for i in range(int(data.shape[0]/BATCH_SIZE + 1)):
        batch = data.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        input = torch.cat((torch.tensor(batch[["log_IMDB", "log_IMDB_nostem", "rating_similarity", "avg_rating", "tag_exists", "lsi_tags_75", "lsi_imdb_175", "tag_prob"]].values.tolist()), torch.tensor(batch["one_hot"].values.tolist())), dim=1).to(device, dtype=torch.float)
        target = ((torch.tensor(batch["targets"].tolist()) - 1) / 4).to(device, dtype=torch.float)         
        optimizer.zero_grad()
        predicted = model(input).to(device, dtype=torch.float).squeeze()
        loss = loss_fn(predicted, target)
        loss.backward()
        optimizer.step()
        if i%100 == 0:
            print(f"epoch {epoch} batch {i} loss: {loss}")

def predict(data):
    model.eval()
    with torch.no_grad():
        output_df = pd.DataFrame(columns=["item_id", "tag", "predictions"])
        for i in range(int(data.shape[0]/BATCH_SIZE + 1)):
            batch = data.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            input = torch.cat((torch.tensor(batch[["log_IMDB", "log_IMDB_nostem", "rating_similarity", "avg_rating", "tag_exists", "lsi_tags_75", "lsi_imdb_175", "tag_prob"]].values.tolist()), torch.tensor(batch["one_hot"].values.tolist())), dim=1).to(device, dtype=torch.float)        
            optimizer.zero_grad()
            predicted = model(input).to(device, dtype=torch.float).squeeze().cpu().tolist()
            batch_df = pd.DataFrame({'item_id': batch['item_id'].tolist(), "tag": batch["tag"].tolist(), 'predictions': predicted})
            output_df = pd.concat([output_df, batch_df])
        output_df.to_csv(f'predictions{TENFOLD}/tagdl_predictions{TENFOLD}.csv', index=False)

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

tag_dict, tag_index = get_tag_indexing()
train_data = pd.read_csv("movie_dataset_public_final/processed/10folds/train0.csv").rename(columns={"movieId": "item_id"})
test_data = pd.read_csv("movie_dataset_public_final/processed/10folds/test0.csv").rename(columns={"movieId": "item_id"})
train_data["one_hot"] = train_data["tag"].apply(lambda t: get_one_hot(t))
test_data["one_hot"] = test_data["tag"].apply(lambda t: get_one_hot(t))

print(train_data.shape)
print(test_data.shape)

model = FFNN(1102, hidden_size = 64)
model.to(device)
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

for epoch in range(EPOCHS):
    train(train_data, epoch)

predict(test_data)
