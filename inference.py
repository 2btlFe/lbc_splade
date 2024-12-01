import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.models.transformer_rep import Splade
import ipdb
from tqdm import tqdm
import pandas as pd 
import pickle
import os
from splade.datasets.datasets import CollectionDatasetPreLoad
from splade.datasets.dataloaders import CollectionDataLoader 
import math
torch.set_printoptions(precision=2)

# set the dir for trained weights
##### v2
# model_type_or_dir = "naver/splade_v2_max"
# model_type_or_dir = "naver/splade_v2_distil"
### v2bis, directly download from Hugging Face
# model_type_or_dir = "naver/splade-cocondenser-selfdistil"
model_type_or_dir = "naver/splade-cocondenser-ensembledistil"

# load dataset
device = 'cuda:0'
batch_size = 64
dataset = CollectionDatasetPreLoad('/workspace/ssd0/byeongcheol/splade/msmarco-full/full_collection', id_style="content_id")
loader = CollectionDataLoader(dataset=dataset, tokenizer_type=model_type_or_dir, max_length=512, batch_size=batch_size, shuffle=False, num_workers=4)

# loading model and tokenizer
device = "cuda:0"
model = Splade(model_type_or_dir, agg="max").to(device)
model.eval()
# tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)

# example document from MS MARCO passage collection (doc_id = 8003157)
# df = pd.read_csv('/workspace/ssd0/byeongcheol/splade/msmarco-full/all_train_queries/all_train_queries/train_queries/raw.tsv', sep='\t')
# df = pd.read_csv('/workspace/ssd0/byeongcheol/splade/msmarco-full/full_collection/raw.tsv', sep='\t')

save_dir = 'doc_emb'
os.makedirs(save_dir, exist_ok=True)

result ={}
num_iter = math.ceil(len(dataset)//batch_size)
num_partition = num_iter // 10
print(f"Total number of iteration is {num_iter, num_partition}")

partition = 0
with torch.no_grad():
    for i, token in enumerate(tqdm(loader)):
        # now compute the document representation
        # (id, doc)

        ids = token.pop("id")
        batch = {k: v.to(device) for k, v in token.items()}
        doc_rep = model(d_kwargs=batch)["d_rep"].squeeze()  # (sparse) doc rep in voc space, shape (30522,)

        # print(f"{id} - {doc_rep}")
        for id, doc in zip(ids, doc_rep):
            result[id] = doc.cpu()
        # result[id] = doc_rep

        if i % num_partition == 0:
            with open(f'{save_dir}/save_doc_{partition}.pkl', 'wb') as f:
                pickle.dump(result, f)
            result = {}
            partition += 1

    #rest
    with open(f'{save_dir}/save_doc_{partition}.pkl', 'wb') as f:
        pickle.dump(result, f)
        
        