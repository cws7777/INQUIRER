import torch
#from transformers import BlipProcessor, BlipForConditionalGeneration#, CLIPProcessor, CLIPModel
#import torchvision.transforms as T
import os
import torch.nn.functional as F
import argparse
import json
import glob
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, Dataset


# Dataset 클래스 정의
class How2QADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def calculate_conditional_probability_with_video_batch(batch, model, tokenizer, device): #(text_in, text_out, model, tokenizer):#model_name="google/flan-t5-xxl"):
    """
    Calculate conditional probability of text_out given text_in and video frames.
    
    Args:
        text_in (str): Input text.
        text_out (str): Output text for which the probability is computed.
        video_feat (str): clip feature of the video file.
        num_frames (int): Number of frames to sample for feature extraction.
    
    Returns:
        float: Conditional probability value.
    """
    model.eval()
    batch_probs = []
    with torch.no_grad():
        #import pdb; pdb.set_trace()
        for i in range(len(batch['text_in'])):
            text_in = batch['text_in'][i]
            text_out = batch["text_out"][i]
            
            input_ids = tokenizer(text_in, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            output_ids = tokenizer(text_out, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            
            outputs = model(input_ids=input_ids, decoder_input_ids=output_ids[:, :-1], labels=output_ids[:, 1:], output_hidden_states=True, return_dict=True)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            # Token-level probabilities
            token_probs = [
                probs[0, i, output_ids[0, i + 1]].item()
                for i in range(output_ids.shape[1] - 1)
            ]
            
            # Sequence-level perplexity
            perplex = torch.log(torch.tensor(token_probs)).mean()
            perplexity = torch.exp(perplex).item()
            batch_probs.append(perplexity)
    
    return batch_probs #perplexity

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process How2QA data with CUDA and batch.')
    parser.add_argument('--train_file', type=str, help="Path to the How2QA train file.", default="../gen_how2qa/train.json")
    parser.add_argument('--add_file', type=str, help="Path to the additional How2QA file.", default="./results_kg_filled.json")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for processing.")
    parser.add_argument('--output_file', type=str, default="./how2qa_train_train_kg_prob.json", help="Output file for results.")
    parser.add_argument('--prompt_path', type=str, default="./prompt_20250511_genKGQA.txt")
    args = parser.parse_args()

    prompt_format = open(args.prompt_path, 'r').read().strip()
    # CUDA device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델과 토크나이저 로드
    model_name = "google/flan-t5-xxl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # 데이터 로드
    with open(args.train_file, 'r') as train_file:
        train_data = json.load(train_file)
    
    with open(args.add_file, 'r') as add_file:
        additional_data = json.load(add_file)

    with open('../../../../Flipped-VQA/data/how2qa/subtitles.pkl', 'rb') as subt_file:
        subtitle_data = pickle.load(subt_file)
    
    with open('./knowledge.json', 'r', encoding='utf-8') as kg_file:
        knowledge_dict = json.load(kg_file)
    
    # 데이터 준비
    dataset = []
    dict_list = []
    for i, item in enumerate(train_data):
        #q1_numb = i * 2

        qid = item['qid'] # type -> string
        question = item['question']
        video_id = item['video_id']
        start_t = item['start']
        end_t = item['end']
        a0 = item['a0']
        a1 = item['a1']
        a2 = item['a2']
        a3 = item['a3']
        answer_id = item['answer_id']

        all_script = subtitle_data[video_id]
        if video_id in knowledge_dict:
            kg = knowledge_dict[video_id]
        else:
            kg = {}

        prompt = prompt_format.format(question=question, qid=qid, a0=a0, a1=a1, a2=a2, a3=a3, 
                                answer_id=answer_id, item=item, video_id=video_id, start_t=start_t, end_t=end_t, all_script=all_script, knowledge_graph=kg)
        text_in = prompt

        text_out_1 = ""
        for j, item_add in enumerate(additional_data):
            # print(type(item_add["qid"])) type -> string
            add_qid = item_add["qid"]
            if qid != add_qid:
                continue
            else:
                try:
                    text_out_1 = item_add["question"]
                except:
                    import pdb; pdb.set_trace()
                dataset.append({"text_in": text_in, "text_out": text_out_1})
                dict_list.append(item_add)


    # DataLoader 준비
    dataset = How2QADataset(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 결과 저장
    with open(args.output_file, 'w') as output_file:
        total_prob = 0.0
        count = 0

        new_list = []
        for batch in dataloader:
            batch_probs = calculate_conditional_probability_with_video_batch(batch, model, tokenizer, device)
            for j in range(len(batch_probs)):
                prob = batch_probs[j]
                add_dict = dict_list[count]
                add_dict["perplex"] = prob
                print(prob)
                new_list.append(add_dict)
                #output_file.write(json.dumps(add_dict) + '\n')
                total_prob += prob
                count += 1
            print(count)
        json.dump(new_list, output_file)
        # 평균 perplexity 저장
        average_prob = total_prob / count
        print(f"Average perplexity: {average_prob}\n")
        print(f"Total questions processed: {count}")
        print(f"Average perplexity: {average_prob}")
