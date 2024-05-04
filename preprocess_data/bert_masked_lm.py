import torch
import numpy as np
import pandas as pd
from torch import cuda
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import BertForMaskedLM, BertConfig
from torch.utils.data import Dataset

TRAIN_BATCH_SIZE = 16
MAX_LEN = 512
EPOCHS = 1
CORPUS_FILE = ""
MODEL_PATH = ""

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if cuda.is_available() else 'cpu'

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.item_ids = dataframe.item_id
        self.text = dataframe.txt
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
        input_ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'item_ids': self.item_ids[index],
        }

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

config = BertConfig(
    vocab_size=tokenizer.vocab_size
)

model = BertForMaskedLM(config) # first iteration
#model = BertForMaskedLM.from_pretrained(MODEL_PATH) # after first iteration, uncomment this to load the model trained so far
model.to(device)

data = pd.read_csv(CORPUS_FILE).sample(frac=1, random_state=42).reset_index(drop=True)

training_set = CustomDataset(data, tokenizer, MAX_LEN)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./mlm",
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_gpu_train_batch_size=TRAIN_BATCH_SIZE,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    seed=42,
    data_seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=training_set,
)

trainer.train()
trainer.save_model(MODEL_PATH)
