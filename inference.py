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

save_dir = 'doc_emb'
os.makedirs(save_dir, exist_ok=True)

result ={}
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
            result[id.item()] = doc.cpu()
        # result[id] = doc_rep

        if i % 2000 == 0:
            with open(f'{save_dir}/save_doc_{partition}.pkl', 'wb') as f:
                pickle.dump(result, f)
            del result
            result = {}
            partition += 1

    #rest
    with open(f'{save_dir}/save_doc_{partition}.pkl', 'wb') as f:
        pickle.dump(result, f)
        
        