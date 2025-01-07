import torch
#from transformers import BlipProcessor, BlipForConditionalGeneration#, CLIPProcessor, CLIPModel
#import torchvision.transforms as T
import os
import torch.nn.functional as F
import argparse
import json
import glob
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, Dataset


# Dataset 클래스 정의
class DramaQADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_scripts(args):
    speech_path = os.path.join(args.root_dir, args.speech_path, 'DramaCap_train_script.json')
    script_list = json.load(open(speech_path, 'r'))
    scripts = {}
    for sample in script_list:
        try:
            vid = sample["vid"]
            # if not vid.endswith('0000'):    continue
            description = sample["desc"]
            subs = ''
            if sample["subtitle"] == ".":
                pass
            else:
                for sub in sample["subtitle"]["contained_subs"]:
                    if subs != '':
                        subs += '\n'
                    subs += f'{sub["speaker"]}: {sub["utter"].strip()}'
                
            scripts[vid] = {
                "subs": subs,
                "description": description,
            }
        except:
            from pprint import pprint
            pprint(sample, width=200)
    
    return scripts

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
    parser = argparse.ArgumentParser(description='Process DramaQA data with CUDA and batch.')
    parser.add_argument('--train_file', type=str, required=True, help="Path to the DramaQA shot train file.", default="./temp_data/AnotherMissOhQA_train_epsc.json")
    parser.add_argument('--add_file', type=str, required=True, help="Path to the additional DramaQA file.", default="./temp_data/AnotherMissOhQA_train_set_naive_sc.json")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for processing.")
    parser.add_argument('--output_file', type=str, default="./AnotherMissOhQA_train_set_naive_sceneprob.json", help="Output file for results.")
    parser.add_argument('--prompt_path', type=str, default="./sceneaddQA_20241122_wo_qckg.txt")
    parser.add_argument('--root_dir', type=str, default="/data1/AnotherMissOh/")
    
    args = parser.parse_args()

    prompt_format = open(args.prompt_path, 'r').read().strip()
    # CUDA device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델과 토크나이저 로드
    model_name = "google/flan-t5-xxl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # 데이터 로드
    #train_data = load_and_merge_jsonrows(args.root_dir + 'addsceneQA_ws/' + args.train_file)
    with open(args.train_file, 'r') as train_file:
        train_data = json.load(train_file)
    
    with open(args.add_file, 'r') as add_file:
        additional_data = json.load(add_file)
    
    # 데이터 준비
    dataset = []
    dict_list = []
    for episode in train_data:
        scene_level = episode["scene_level"]
        for scene in scene_level:
            for i, (k, v) in enumerate(scene.items()):
                scene_id = k
                scene_description = v['scene_description']
                #scene_kg = v['scene_knowledge']
                if 'scene_qas' not in v:
                    scene_qas = []
                else:
                    scene_qas = v['scene_qas']

                prompt = prompt_format.format(scene_description=scene_description, scene_QA=scene_qas)
                text_in = prompt
                count = 0
                for i in range(len(additional_data)):
                    if scene_id != additional_data[i]["vid"]:
                        continue
                    else:
                        text_out_1 = additional_data[i]["que"]
                        dataset.append({"text_in": text_in, "text_out": text_out_1})
                        dict_list.append(additional_data[i])
        #dict_list.append(additional_data[q1_numb+1])

    # DataLoader 준비
    dataset = DramaQADataset(dataset)
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
                new_list.append(add_dict)
                #output_file.write(json.dumps(add_dict) + '\n')
                total_prob += prob
                count += 1
            print(count)
        json.dump(new_list, output_file, indent=4)
        # 평균 perplexity 저장
        average_prob = total_prob / count
        print(f"Average perplexity: {average_prob}\n")
        print(f"Total questions processed: {count}")
        print(f"Average perplexity: {average_prob}")