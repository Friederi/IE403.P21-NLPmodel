# %% [markdown]
# # Library import

# %%
import torch


from transformers import BertModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class BertWithScalarFeatures(nn.Module):
    def __init__(self, scalar_feature_dim, num_classes):
        super(BertWithScalarFeatures, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.bert.config.hidden_size + scalar_feature_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask, scalar_features):
        # BERT output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        combined_features = torch.cat((cls_embedding, scalar_features), dim=1)
        x = F.relu(self.fc1(combined_features))
        x = self.fc2(x)

        return x

    def predict_proba(self, input_ids, attention_mask, scalar_features):
        logits = self.forward(input_ids, attention_mask, scalar_features)
        
        probabilities = F.softmax(logits, dim=1)
        return probabilities
    
class CommentDataset(Dataset):
    def __init__(self, comments, scalar_features, labels, tokenizer, max_len=128):
        self.comments = comments
        self.scalar_features = scalar_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = self.comments[idx]
        scalar_feat = torch.tensor(self.scalar_features[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        encoded = self.tokenizer(
            comment,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'scalar_features': scalar_feat,
            'label': label
        }
