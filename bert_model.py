import numpy as np
import pandas as pd
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from json import loads
from torch import cuda
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if cuda.is_available() else 'cpu'

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 1e-05
TENFOLD = 1
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.item_ids = dataframe.item_id
        self.text = dataframe.txt
        self.targets = dataframe.targets
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'item_ids': self.item_ids[index],
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, ids, mask, token_type_ids):
        output_1 = self.bert(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.dropout(output_1.pooler_output)
        output = self.linear(output_2)
        output = self.sigmoid(output)
        return output

def train(epoch):
    epoch_loss = 0
    model.train()
    for batch ,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)
        outputs = model(ids, mask, token_type_ids).to(device, dtype= torch.float)
        optimizer.zero_grad() 
        loss = loss_fn(outputs, targets)
        epoch_loss += loss
        if batch%50==0:
            print(f'Batch: {batch}, Loss:  {loss.item()}')
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch}: Train loss: {epoch_loss / len(training_loader)}')

def test(file_ending, TENFOLD):
    model.eval()
    with torch.no_grad():
        output_df = pd.DataFrame(columns=['item_id', 'predictions'])
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids).to(device, dtype = torch.float)
            batch_df = pd.DataFrame({'item_id': data['item_ids'].tolist(), 'predictions': outputs.tolist()})
            output_df = pd.concat([output_df, batch_df])
        output_df.to_csv(f'predictions{TENFOLD}/predictions{TENFOLD}_tagid_{file_ending}', index=False)

with open (f"review_list{TENFOLD}.txt", "r") as file:
    review_list = file.read().split("\n")[:-1]

for i in range(len(review_list)):
    print(f"tenfold {TENFOLD} file {i}")
    test_file = review_list[i]
    train_file = test_file.replace("test", "train")
    file_ending = test_file.split("_")[4]
    print(test_file)

    test_data = pd.read_csv(f"reviews/{TENFOLD}/{test_file}")
    test_data.targets = test_data.targets.map(lambda r: np.array([r]))
    print("test shape:", test_data.shape)
    train_data = pd.read_csv(f"reviews/{TENFOLD}/{train_file}")
    train_data.targets = train_data.targets.map(lambda r: np.array([r]))
    print("train shape:", train_data.shape)

    training_set = CustomDataset(train_data, tokenizer, MAX_LEN)
    testing_set = CustomDataset(test_data, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                }

    test_params = {'batch_size': TEST_BATCH_SIZE,
                    'shuffle': False,
                }

    testing_loader = DataLoader(testing_set, **test_params)
    training_loader = DataLoader(training_set, **train_params)

    model = BERTClass()
    model.to(device)
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(params =  model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train(epoch)
    test(file_ending, TENFOLD)

