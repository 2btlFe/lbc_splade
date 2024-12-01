import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.models.transformer_rep import Splade
import ipdb
from tqdm import tqdm
import pickle

with open('/workspace/ssd0/byeongcheol/splade/doc_emb/save_doc_1.pkl', 'rb') as f:
    data = pickle.load(f)

ipdb.set_trace()