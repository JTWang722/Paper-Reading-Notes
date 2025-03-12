from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import random
import wandb
import os


class PathDataset(Dataset):
    def __init__(self, data_path, model_path, max_length):
        self.max_legnth = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.lines = open(data_path, "r").readlines()

        self.ent2text = {}    # 类型 名称
        with open("/data/jtwang/trytrytryllm/datasets/kg-bert/cn15k/entity2text.txt", 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    self.ent2text[temp[0]] = temp[1]
                    print('ent_text', temp[1])

        self.rel2text = {}
        with open("/data/jtwang/trytrytryllm/datasets/kg-bert/cn15k/relation2text.txt", 'r') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    self.rel2text[temp[0]] = temp[1] 
                    print('rel_text', temp[1])
        

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        # score, question, context = self.lines[index].strip("\n").split("\t")
        # input_txt = f"[CLS] {question} [SEP] {context} [SEP]"

        h, r, t, score = self.lines[index].strip("\n").split("\t")
        h_text, r_text, t_text = self.ent2text[h], self.rel2text[r], self.ent2text[t]
        input_txt = f"[CLS] {h_text} [SEP] {r_text} [SEP] {t_text}. [SEP]"
        encoding = self.tokenizer.encode_plus(
            input_txt,
            max_length=self.max_legnth,
            add_special_tokens=False,  # [CLS]和[SEP]
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",  # Pytorch tensor张量
            truncation=True,
        )

        return {
            "input_txt": input_txt,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "toeken_type_ids": encoding["token_type_ids"].flatten(),
            "labels": torch.tensor(eval(score), dtype=torch.float),
        }


class ConfModel(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.d_model = self.bert.pooler.dense.out_features
        self.linear1 = nn.Linear(self.d_model, self.d_model*4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.d_model*4,1)
        self.sigmoid = nn.Sigmoid()
        self.device = device
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)

    def self_attn_encode(self, q, k, v, mask):
        # q: batch_size,1,d_model
        # k: batch_size,seq_len,d_model -> batch_size,d_model,seq_len
        # v: batch_size,seq_len,d_model
        q = self.w_q(q)
        k = self.w_k(k).transpose(1, 2)
        v = self.w_v(v)

        # attn_logit: batch_size,1,seq_len
        attn_logit = torch.matmul(q, k)
        if mask != None:
            attn_logit = attn_logit.masked_fill(mask == 0, -1e9)
        attn_weight = torch.softmax(attn_logit, dim=-1)
        # batch_size,1,seq_len * batch_size,seq_len,d_model -> batch_size,1,d_model
        weight_sum = torch.matmul(attn_weight, v)
        output = self.w_o(weight_sum)
        return output

    def forward(self, x, mode):
        input_ids = x["input_ids"].to(self.device)
        attn_mask = x["attention_mask"].to(self.device)

        output = self.bert(input_ids=input_ids, attention_mask=attn_mask)

        last_hidden_state = output[
            "last_hidden_state"
        ]  # batch_size*max_token_length*d_model
        pooler_output = output["pooler_output"]  # batch_size,d_model

        if mode == "CLS":
            score = pooler_output
        elif mode == "avg":
            extended_attn_mask = (
                attn_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
            )
            last_hidden_state_masked = last_hidden_state * extended_attn_mask
            sum_embeddings = torch.sum(last_hidden_state_masked, dim=1)
            sum_mask = torch.sum(extended_attn_mask, dim=1)
            mean_pooled = sum_embeddings / sum_mask
            score = mean_pooled
        elif mode == "attn":
            output = self.self_attn_encode(
                last_hidden_state[:, 0, :].unsqueeze(1),
                last_hidden_state,
                last_hidden_state,
                attn_mask.unsqueeze(1),
            )
            score = output
        score = self.linear2(self.relu(self.linear1(score))).flatten()
        return score


def train_model(model, dataloader, criterion, optimizer, scheduler, device, strategy, output_f):
    model = model.to(device).train()
    criterion = criterion.to(device)

    for i, batch in enumerate(dataloader):
        # if i < 3:
        #     print(batch)
        input_txt = batch["input_txt"]
        labels = batch["labels"].to(device)
        output = model(batch, strategy)

        loss = criterion(output, labels)

        if i % 50 == 0:
            with torch.no_grad():
                diff = output - labels
                abs_diff = torch.abs(diff)
                mae = torch.mean(abs_diff)
            print(f"{i}\tMSE:{loss.item()}\tMAE:{mae}")
            print(f"{i}\tMSE:{loss.item()}\tMAE:{mae}", file=output_f)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


def test_model(model, dataloader, criterion, device, mode, strategy, output_f):
    with torch.no_grad():
        output_list = None
        label_list = None
        model = model.to(device).eval()
        for i, batch in enumerate(dataloader):
            input_txt = batch["input_txt"]
            labels = batch["labels"].to(device)
            output = model(batch, strategy)
            if output_list == None:
                output_list = output
            else:
                output_list = torch.cat((output_list, output), dim=0)
            if label_list == None:
                label_list = labels
            else:
                label_list = torch.cat((label_list, labels), dim=0)

            diff = output_list - label_list
            abs_diff = torch.abs(diff)
            mae = torch.mean(abs_diff)
            mse = criterion(output_list, label_list).item()
        print(f"{mode}\tMSE: {mse}\tMAE: {mae}")
        print(f"{mode}\tMSE: {mse}\tMAE: {mae}", file=output_f)



def main(data_path, model_path, strategy, result_path):
    # wandb.init()
    EXP_name = ''
    

    batch_size = 256
    max_length = 30
    train_dataset = PathDataset(data_path + "train.tsv", model_path, max_length)
    valid_dataset = PathDataset(data_path + "val.tsv", model_path, max_length)
    test_dataset = PathDataset(data_path + "test.tsv", model_path, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0")

    myModel = ConfModel(model_path, device)

    EPOCHS = 100
    criterion = torch.nn.MSELoss()
    optimizer = AdamW(myModel.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )
    with open(result_path, 'w') as fw:
        for e in range(EPOCHS):
            print(f"Epoch: {e}")
            print(f"Epoch: {e}", file=fw)
            train_model(myModel, train_dataloader, criterion, optimizer, scheduler, device, strategy, fw)
            # torch.save(myModel.state_dict(), f"{EXP_name}/model_{e}.pt")
            test_model(myModel, valid_dataloader, criterion, device, "valid", strategy, fw)
            test_model(myModel, test_dataloader, criterion, device, "test", strategy, fw)


if __name__ == "__main__":

    data_path = "/data/jtwang/trytrytryllm/datasets/kg-bert/cn15k/"
    model_path = "/data/jtwang/trytrytryllm/kg-bert/bert-base-uncased"
    strategy = "CLS"
    result_path = "/data/jtwang/trytrytryllm/kg-bert/cyl_output/cn15k/cls_0111.txt"

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    main(data_path, model_path, strategy, result_path)
