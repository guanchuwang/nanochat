import json
import metrics
# import nltk

# try:
#     nltk.data.find("tokenizers/punkt_tab")
# except LookupError:
#     nltk.download('punkt_tab')
    
    
def nlp_eval_acc(answer_file):
    data_name = answer_file.split("/")[-1].split("_")[-1].split(".")[0]
    model_name = '_'.join(answer_file.split("/")[-1].split("_")[:-1])
    predictions = []
    references = []
    
    with open(answer_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            output = data['generation']
            gt_ans = data['gt_answer'].split("\n####")[-1].strip()
            
            predictions.append(output)
            references.append(gt_ans)
    
    em_score = metrics.calculate_em(predictions, references)
    f1_score = metrics.calculate_f1_score(predictions, references)
    
    print(f"{model_name} Results on {data_name}:")
    print(f"Exact Match Accuracy: {100*em_score:.3f}%")
    print(f"F1 Score: {100*f1_score:.3f}%")
    return em_score, f1_score
