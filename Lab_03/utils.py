import torch
from transformers import BertTokenizer
from torchmetrics.text import BLEUScore

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BOS_IDX = 101
EOS_IDX = 102
PAD_IDX = 0

def tokenizer_chinese(): 
    tokenizer_cn = BertTokenizer.from_pretrained("bert-base-chinese")
    return tokenizer_cn


def tokenizer_english(): 
    tokenizer_en = BertTokenizer.from_pretrained("bert-base-cased")
    return tokenizer_en


def bleu_score_func(predicted: str, truth: str, grams=1): 
    preds = [predicted]
    truth = [[truth]]
    bleu = BLEUScore(n_gram=grams)
    return bleu(preds, truth)


def BLEU_batch(predict: torch.Tensor, truth: torch.Tensor, output_tokenizer): 
    batch_size = predict.size(0)
    seq_len = (truth == EOS_IDX).nonzero(as_tuple=True)[1] + 1

    total_score = 0
    for i in range(batch_size): 
        predict_str = output_tokenizer.decode(predict[i, :seq_len[i]], skip_special_tokens=True)
        truth_str = output_tokenizer.decode(truth[i, :], skip_special_tokens=True)        
        score_gram1 = bleu_score_func(predict_str.lower(), truth_str.lower(), grams=1)
        #score_gram2 = bleu_score_func(predict_str.lower(), truth_str, grams=2)
        #score_gram3 = bleu_score_func(predict_str.lower(), truth_str, grams=3)
        #score_gram4 = bleu_score_func(predict_str.lower(), truth_str, grams=4)
        #total_score = total_score + (score_gram1 + score_gram2 + score_gram3 + score_gram4) / 4.0        
        total_score = total_score + score_gram1
    total_score = total_score / batch_size
    return total_score