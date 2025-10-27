import pandas as pd
import math
import sys
from timeit import default_timer as timer
from utils import *
from network import *

def main():
    if len(sys.argv) >= 2: 
        MODEL_PATH = sys.argv[1]
    else: 
        MODEL_PATH = "./model.ckpt"
    if len(sys.argv) >= 3:
        DATA_PATH = sys.argv[2]
    else: 
        DATA_PATH = "./translation_test_data.json"

    # Load model
    model = load_model(MODEL_PATH)
    model.to(DEVICE)
    param_model = sum(p.numel() for p in model.parameters())
    print (f"The parameter size of model is {param_model/1000} k")
    # check parameter size requirement
    if(param_model / 1000 > 100000): 
        print("\033[31m====================  FAIL parameter size requirement  ====================\033[0m")
    else: 
        print("\033[32m====================  PASS parameter size requirement  ====================\033[0m")

    # Load testing data and tokenizer
    translation_data = pd.read_json(DATA_PATH)
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

        #print(f"--- Data {i+1}")
        #print("ground truth: ", ground_truth)
        #print("predict:      ", predict)
        #print("score (1-gram) = ", score_gram1)
    end_time = timer()
    execution_time = end_time - start_time
    print("BLEU score (1-gram) = ", score_final_gram1 / len(translation_data))
    print("BLEU score (2-gram) = ", score_final_gram2 / len(translation_data))
    print("BLEU score (3-gram) = ", score_final_gram3 / len(translation_data))
    print("BLEU score (4-gram) = ", score_final_gram4 / len(translation_data))
    # check BLEU score requirement
    if(score_final_gram1 / len(translation_data) < 0.25 or score_final_gram2 / len(translation_data) < 0.1): 
        print("\033[31m====================  FAIL BLEU score requirement      ====================\033[0m")
    else: 
        print("\033[32m====================  PASS BLEU score requirement      ====================\033[0m")

    print(f"execution time = {execution_time:.3f}s")
    # check program execution time requirement  
    if(execution_time > 200.0): 
        print("\033[31m====================  FAIL execution time requirement  ====================\033[0m")
    else: 
        print("\033[32m====================  PASS execution time requirement  ====================\033[0m")

if __name__ == '__main__':
    main()