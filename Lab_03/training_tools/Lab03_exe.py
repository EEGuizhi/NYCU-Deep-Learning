# NYCU IEE Deep Learning Lab 03: Machine Translation
# BSChen (313510156)

# ------------------------------------------------------------------------------ #
# Settings
# ------------------------------------------------------------------------------ #
# Basic settings
LOG_TERMINAL_OUTPUTS_FILE = "terminal_outputs_log.txt"
LOG_FILE_PATH = "log.csv"
SEED = 29

# Training settings
BATCH_SIZE = 8
NUM_EPOCHS = 80
SAVE_TOLERANCE = 0.0005
LEARNING_RATE = 1e-2

# Path settings
TRAIN_PATH = "./translation_train_data.json"
TEST_PATH = "./translation_test_data.json"
MODEL_SAVE_PATH = "./model.ckpt"

def print_and_log(*args, **kwargs):
    print(*args, **kwargs)
    with open(LOG_TERMINAL_OUTPUTS_FILE, "a") as f:
        print(*args, **kwargs, file=f)

# ------------------------------------------------------------------------------ #
# Import Libraries
# ------------------------------------------------------------------------------ #
import os
import json
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from timeit import default_timer as timer

from utils import *
from network import *

# added by myself
import time
from tqdm import tqdm

# Clear the log file at the start
with open(LOG_TERMINAL_OUTPUTS_FILE, "w") as f:
    f.write(f"Start time: {time.ctime()}\n")

# ------------------------------------------------------------------------------ #
# Set Random Seed
# ------------------------------------------------------------------------------ #
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(SEED)
print_and_log(f">> Random seed set to {SEED}\n")


# ------------------------------------------------------------------------------ #
# Data Preprocessing & Tokenization
# ------------------------------------------------------------------------------ #
data_dir = TRAIN_PATH
# BATCH_SIZE = 8

translation_raw_data = pd.read_json(data_dir)
tokenizer_en = tokenizer_english()
tokenizer_cn = tokenizer_chinese()

english_seqs = translation_raw_data["English"].apply(lambda x: tokenizer_en.encode(x, add_special_tokens=True, padding=False))
chinese_seqs = translation_raw_data["Chinese"].apply(lambda x: tokenizer_cn.encode(x, add_special_tokens=True, padding=False))

MAX_TOKENIZE_LENGTH = max(english_seqs.str.len().max(), chinese_seqs.str.len().max()) # longest string
MAX_TOKENIZE_LENGTH = pow(2, math.ceil(math.log(MAX_TOKENIZE_LENGTH) / math.log(2)))  # closest upper to the power of 2

print_and_log(f"Max tokenize length: {MAX_TOKENIZE_LENGTH}")


# ------------------------------------------------------------------------------ #
# Add Padding
# ------------------------------------------------------------------------------ #
def add_padding(token_list: list, max_length: int) -> list:
    if len(token_list) < max_length:
        padding_length = max_length - len(token_list)
        token_list = token_list + [PAD_IDX] * padding_length
    else:
        token_list = token_list[:max_length]  # Trim to MAX_LENGTH if longer
    return token_list

chinese_seqs = chinese_seqs.apply(lambda x: add_padding(x, MAX_TOKENIZE_LENGTH))
english_seqs = english_seqs.apply(lambda x: add_padding(x, MAX_TOKENIZE_LENGTH))

# Check the padding result
print_and_log("===== Chinese tokenized data =====")
print_and_log(f"{chinese_seqs.iloc[0]}\n")

print_and_log("===== English tokenized data =====")
print_and_log(f"{english_seqs.iloc[0]}\n")

# ------------------------------------------------------------------------------ #
# DataLoader
# ------------------------------------------------------------------------------ #
data_size  = len(translation_raw_data)
train_size = int(0.95 * data_size)
valid_size = data_size - train_size
print_and_log("train size:", train_size)
print_and_log("valid size:", valid_size)

en_train_data = []
cn_train_data = []
en_valid_data = []
cn_valid_data = []

for i in range(data_size):
    if (i < train_size):
        en_train_data.append(torch.Tensor(english_seqs.iloc[i]))
        cn_train_data.append(torch.Tensor(chinese_seqs.iloc[i]))
    else:
        en_valid_data.append(torch.Tensor(english_seqs.iloc[i]))
        cn_valid_data.append(torch.Tensor(chinese_seqs.iloc[i]))

class TextTranslationDataset(Dataset): 
    def __init__(self, src, dst, augment_prob=0):
        self.src_list = src
        self.dst_list = dst
        self.augment_prob = augment_prob

    def __len__(self): 
        return len(self.src_list)

    def __getitem__(self, idx):
        if random.random() < self.augment_prob and self.src_list[idx].shape[0] > 7:
            # Random dropout
            drop_num = random.randint(1, self.src_list[idx].shape[0] - 1)
            src = self.src_list[idx].clone()
            src[drop_num] = PAD_IDX
        else:
            src = self.src_list[idx]
        return src, self.dst_list[idx]

cn_to_en_train_set = TextTranslationDataset(cn_train_data, en_train_data, augment_prob=0.3)
cn_to_en_valid_set = TextTranslationDataset(cn_valid_data, en_valid_data)

cn_to_en_train_loader = DataLoader(cn_to_en_train_set, batch_size=BATCH_SIZE, shuffle=False)
cn_to_en_valid_loader = DataLoader(cn_to_en_valid_set, batch_size=BATCH_SIZE, shuffle=True)


# ------------------------------------------------------------------------------ #
# Model
# ------------------------------------------------------------------------------ #
model = load_model()

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

param_model = sum(p.numel() for p in model.parameters())
print_and_log(f"The parameter size of model is {param_model / 1000} k\n")


# ------------------------------------------------------------------------------ #
# Training
# ------------------------------------------------------------------------------ #
# NUM_EPOCHS = 20
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)  # label_smoothing=0.1
# optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-5)
# scheduler = None
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-5
)

def train_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_dataloader: DataLoader):
    model.train()
    losses = 0

    for src, tgt in tqdm(train_dataloader):
        # src, tgt shape: (batch_size, seq_length)
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input  = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)

        optimizer.zero_grad()
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1).long())
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))


def evaluate(model: torch.nn.Module, val_dataloader: DataLoader):
    model.eval()
    losses = 0
    score = 0

    for src, tgt in tqdm(val_dataloader):
        # src, tgt shape: (batch_size, seq_length)
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        logits = model(src, tgt_input)
        _, tgt_predict = torch.max(logits, dim=-1)
        score_batch = BLEU_batch(tgt_predict, tgt_output, tokenizer_en)

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1).long())
        losses += loss.item()
        score += score_batch

    return (losses / len(list(val_dataloader))), (score / len(list(val_dataloader)))


# ------------------------------------------------------------------------------ #
# Start Training
# ------------------------------------------------------------------------------ #
# MODEL_SAVE_PATH = "./model.ckpt"
# LOG_FILE_PATH = "log.csv"
# SAVE_TOLERANCE = 0.01

print_and_log("\n>> Start training...")
print_and_log(f"Model training on device: {DEVICE}")

model = model.to(DEVICE)
with open(LOG_FILE_PATH, 'w') as log_file:
    log_file.write("Epoch, Train Loss, Val Loss, Val Acc, Epoch Time\n")

best_acc = 0
for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer, cn_to_en_train_loader)
    end_time = timer()
    val_loss, val_acc = evaluate(model, cn_to_en_valid_loader)

    # Log the results
    print_and_log((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    with open(LOG_FILE_PATH, 'a') as log_file:
        log_file.write(f"{epoch}, {train_loss:.3f}, {val_loss:.3f}, {val_acc:.3f}, {(end_time - start_time):.3f}\n")

    # Save the best model so far.
    if val_acc > best_acc - SAVE_TOLERANCE:
        best_acc = val_acc if val_acc > best_acc else best_acc
        best_state_dict = model.state_dict()
        torch.save(best_state_dict, MODEL_SAVE_PATH)
        print_and_log("(model saved)")

    if scheduler:
        scheduler.step(val_loss)

print_and_log(">> Training complete.\n")


# ------------------------------------------------------------------------------ #
# Inference
# ------------------------------------------------------------------------------ #
# Load the best model
model = load_model(MODEL_PATH="model.ckpt")
model = model.to(DEVICE)

# Stimulus 1
sentence = "你好，欢迎来到中国。"
ground_truth = 'Hello, welcome to China.'
predicted = translate(model, sentence, tokenizer_cn, tokenizer_en)
print_and_log(f'{"Input:":15s}: {sentence}')
print_and_log(f'{"Prediction":15s}: {predicted}')
print_and_log(f'{"Ground truth":15s}: {ground_truth}')
print_and_log("Bleu Score (1-gram): ", bleu_score_func(predicted.lower(), ground_truth.lower(), 1).item())
print_and_log("Bleu Score (2-gram): ", bleu_score_func(predicted.lower(), ground_truth.lower(), 2).item())
print_and_log("Bleu Score (3-gram): ", bleu_score_func(predicted.lower(), ground_truth.lower(), 3).item())
print_and_log("Bleu Score (4-gram): ", bleu_score_func(predicted.lower(), ground_truth.lower(), 4).item())
print_and_log("")

# Stimulus 2
sentence = "她知道您的電話號碼嗎?"
ground_truth = 'Does she know your telephone number?'
predicted = translate(model, sentence, tokenizer_cn, tokenizer_en)
print_and_log(f'{"Input:":15s}: {sentence}')
print_and_log(f'{"Prediction":15s}: {predicted}')
print_and_log(f'{"Ground truth":15s}: {ground_truth}')
print_and_log("Bleu Score (1-gram): ", bleu_score_func(predicted.lower(), ground_truth.lower(), 1).item())
print_and_log("Bleu Score (2-gram): ", bleu_score_func(predicted.lower(), ground_truth.lower(), 2).item())
print_and_log("Bleu Score (3-gram): ", bleu_score_func(predicted.lower(), ground_truth.lower(), 3).item())
print_and_log("Bleu Score (4-gram): ", bleu_score_func(predicted.lower(), ground_truth.lower(), 4).item())
print_and_log("")

# Stimulus 3
sentence = "你现在在哪里工作?"
ground_truth = 'Where do you work now?'
predicted = translate(model, sentence, tokenizer_cn, tokenizer_en)
print_and_log(f'{"Input:":15s}: {sentence}')
print_and_log(f'{"Prediction":15s}: {predicted}')
print_and_log(f'{"Ground truth":15s}: {ground_truth}')
print_and_log("Bleu Score (1-gram): ", bleu_score_func(predicted.lower(), ground_truth.lower(), 1).item())
print_and_log("Bleu Score (2-gram): ", bleu_score_func(predicted.lower(), ground_truth.lower(), 2).item())
print_and_log("Bleu Score (3-gram): ", bleu_score_func(predicted.lower(), ground_truth.lower(), 3).item())
print_and_log("Bleu Score (4-gram): ", bleu_score_func(predicted.lower(), ground_truth.lower(), 4).item())
print_and_log("")

# ------------------------------------------------------------------------------ #
# Testing
# ------------------------------------------------------------------------------ #
print_and_log("\n>> Start testing...")

# Load model
# model = load_model(MODEL_SAVE_PATH)
# model.to(DEVICE)
param_model = sum(p.numel() for p in model.parameters())
print_and_log(f"The parameter size of model is {param_model / 1000} k")

# Check parameter size requirement
if(param_model / 1000 > 100000): 
    print_and_log("\033[31m====================  FAIL parameter size requirement  ====================\033[0m")
else: 
    print_and_log("\033[32m====================  PASS parameter size requirement  ====================\033[0m")

# Load testing data and tokenizer
translation_data = pd.read_json(TEST_PATH)
tokenizer_en = tokenizer_english()
tokenizer_cn = tokenizer_chinese()

score_final_gram1 = 0
score_final_gram2 = 0
score_final_gram3 = 0
score_final_gram4 = 0
start_time = timer()
for i in range(len(translation_data)): 
    sentence = translation_data["Chinese"].iloc[i]
    ground_truth = translation_data["English"].iloc[i]
    predict = translate(model, sentence, tokenizer_cn, tokenizer_en)

    score_gram1 = bleu_score_func(predict.lower(), ground_truth.lower(), 1).item()
    score_gram2 = bleu_score_func(predict.lower(), ground_truth.lower(), 2).item()
    score_gram3 = bleu_score_func(predict.lower(), ground_truth.lower(), 3).item()
    score_gram4 = bleu_score_func(predict.lower(), ground_truth.lower(), 4).item()
    score_final_gram1 += score_gram1
    score_final_gram2 += score_gram2
    score_final_gram3 += score_gram3
    score_final_gram4 += score_gram4

    # print(f"--- Data {i+1}")
    # print("ground truth: ", ground_truth)
    # print("predict:      ", predict)
    # print("score (1-gram) = ", score_gram1)
end_time = timer()
execution_time = end_time - start_time
print_and_log("BLEU score (1-gram) = ", score_final_gram1 / len(translation_data))
print_and_log("BLEU score (2-gram) = ", score_final_gram2 / len(translation_data))
print_and_log("BLEU score (3-gram) = ", score_final_gram3 / len(translation_data))
print_and_log("BLEU score (4-gram) = ", score_final_gram4 / len(translation_data))
# check BLEU score requirement
if(score_final_gram1 / len(translation_data) < 0.25 or score_final_gram2 / len(translation_data) < 0.1): 
    print_and_log("\033[31m====================  FAIL BLEU score requirement      ====================\033[0m")
else: 
    print_and_log("\033[32m====================  PASS BLEU score requirement      ====================\033[0m")

print_and_log(f"execution time = {execution_time:.3f}s")
# check program execution time requirement  
if(execution_time > 200.0): 
    print_and_log("\033[31m====================  FAIL execution time requirement  ====================\033[0m")
else: 
    print_and_log("\033[32m====================  PASS execution time requirement  ====================\033[0m")
