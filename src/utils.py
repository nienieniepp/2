import json, random
from pathlib import Path
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_dirs(*paths):
    for p in paths: Path(p).mkdir(parents=True, exist_ok=True)

def save_json(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f: json.dump(data,f,indent=2,ensure_ascii=False)
