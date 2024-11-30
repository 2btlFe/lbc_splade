import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.models.transformer_rep import Splade
import ipdb
from tqdm import tqdm
import pandas as pd 
import pickle


# set the dir for trained weights
##### v2
# model_type_or_dir = "naver/splade_v2_max"
# model_type_or_dir = "naver/splade_v2_distil"
### v2bis, directly download from Hugging Face
# model_type_or_dir = "naver/splade-cocondenser-selfdistil"
model_type_or_dir = "naver/splade-cocondenser-ensembledistil"


# ipdb.set_trace()

# loading model and tokenizer
model = Splade(model_type_or_dir, agg="max")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

# example document from MS MARCO passage collection (doc_id = 8003157)
df = pd.read_csv('/workspace/ssd0/byeongcheol/splade/msmarco-full/all_train_queries/all_train_queries/train_queries/raw.tsv', sep='\t')

result ={}
for i, (idx, doc) in enumerate(tqdm(df.values)):
    # now compute the document representation
    with torch.no_grad():
        doc_rep = model(d_kwargs=tokenizer(doc, return_tensors="pt"))["d_rep"].squeeze()  # (sparse) doc rep in voc space, shape (30522,)
    
    print(f"{idx} - {doc_rep}")
    result[idx] = doc_rep

    i += 1
    if i % 10000 == 0:
    with open(f'save_doc_{i}.pkl', 'wb') as f:
        pickle.dump(result, f)

