import pandas as pd
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch import cuda
import numpy as np

import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if cuda.is_available() else 'cpu'

MAX_LEN = 16
BATCH_SIZE = 8
TITLE_FILE = ""
OUTPUT_FILE = ""
MODEL_PATH = ""

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.item_ids = dataframe.item_id
        self.text = dataframe.title
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
        }
    
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(MODEL_PATH)
    
    def forward(self, ids, mask, token_type_ids):
        return self.bert(ids, attention_mask = mask, token_type_ids = token_type_ids).last_hidden_state

def create_embeddings():
    model.eval()
    with torch.no_grad():
        output_df = pd.DataFrame(columns=['item_id', 'embedding'])
        for i, data in enumerate(loader, 0):
            if i % 100 == 0:
                print("Batch", i)
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            embeddings = model(ids, mask, token_type_ids)
            mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            mask_sum = mask_expanded.sum(1)
            mask_sum = torch.clamp(mask_sum, min=1e-9)
            embeddings = sum_embeddings / mask_sum
            batch_df = pd.DataFrame({'item_id': data['item_ids'].tolist(), "embedding": embeddings.tolist()})
            output_df = pd.concat([output_df, batch_df])
        output_df.to_csv(OUTPUT_FILE, index=False)

review_file = pd.read_csv(TITLE_FILE)
review_set = CustomDataset(review_file, tokenizer, MAX_LEN)
params = {"batch_size": BATCH_SIZE, "shuffle": False}
loader = DataLoader(review_set, **params)
    
model = BERTClass()
model.to(device)
create_embeddings()

