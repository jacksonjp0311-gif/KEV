from __future__ import annotations
import hashlib,json,re
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE=('COPY_VALUE','SUPERSEDES','MAGNITUDE','NEGATE','REFERENCE','ACTIVE_SELECTION','RUN_STATUS','RECEIPT_VALUE','EVIDENCE_CONSISTENCY')
NEW=('GOAL','CONSTRAINT','OBSERVATION','PREDICTION')
INTENTS=BASE+NEW
D=1024
EMB=80

CANON={
'goal':'GOAL','objective':'GOAL','aim':'GOAL','target':'GOAL','desired':'GOAL','want':'GOAL',
'constraint':'CONSTRAINT','requirement':'CONSTRAINT','must':'CONSTRAINT','cannot':'CONSTRAINT','forbidden':'CONSTRAINT','limit':'CONSTRAINT',
'observed':'OBSERVATION','observation':'OBSERVATION','measured':'OBSERVATION','actual':'OBSERVATION','recorded':'OBSERVATION',
'predict':'PREDICTION','prediction':'PREDICTION','expect':'PREDICTION','forecast':'PREDICTION','anticipate':'PREDICTION','likely':'PREDICTION',
'old':'OLD','former':'OLD','obsolete':'OLD','replace':'UPDATE','replaced':'UPDATE','superseded':'UPDATE','current':'CURRENT','now':'CURRENT',
'absolute':'MAGNITUDE','magnitude':'MAGNITUDE','distance':'MAGNITUDE','negate':'NEGATE','inverse':'NEGATE','opposite':'NEGATE',
'active':'ACTIVE','serving':'ACTIVE','inactive':'INACTIVE','candidate':'INACTIVE','receipt':'RECEIPT','result':'RESULT','output':'RESULT'
}

def _h(s):return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4],'big')%D

def tokens(text):
    out=[]
    for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*|-?\d+",text.casefold()):
        if re.fullmatch(r'-?\d+',w):out.append('NUMBER')
        else:out.append(CANON.get(w,w))
    return out

def features(text):
    t=tokens(text);x=torch.zeros(D)
    for w in t:x[_h('u:'+w)]+=1
    for a,b in zip(t,t[1:]):x[_h('b:'+a+'_'+b)]+=1
    return torch.log1p(x)

class SemanticBreadth(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(D,256),nn.GELU(),nn.LayerNorm(256),nn.Linear(256,EMB))
        self.head=nn.Linear(EMB,len(INTENTS))
    def forward(self,x):
        z=F.normalize(self.enc(x),dim=-1)
        return z,self.head(z)
    @property
    def parameter_count(self):return sum(p.numel() for p in self.parameters())

def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    m=SemanticBreadth();m.load_state_dict(ck['state_dict']);m.eval();return m

def predict(m,text):
    with torch.no_grad():
        z,l=m(features(text)[None,:]);p=torch.softmax(l,1)[0];i=int(p.argmax())
    return {'intent':INTENTS[i],'score':float(p[i]),'embedding':z[0].tolist()}

def train(parent_path,out_dir,steps=800,seed=51401,extra_jsonl=None):
    """Public API contract.

    The canonical research archive contains the authored curriculum and exact checkpoint
    lineage used for published measurements. The GitHub source tree intentionally excludes
    large/generated training artifacts. Supply reviewed JSONL examples and a parent checkpoint
    when reproducing or extending the model.
    """
    if not extra_jsonl:
        raise RuntimeError('Reviewed JSONL training examples are required in the public source build.')
    rows=[]
    for line in Path(extra_jsonl).read_text(encoding='utf-8').splitlines():
        if line.strip():
            r=json.loads(line)
            if r.get('intent') not in INTENTS:raise ValueError('unknown intent')
            rows.append(r)
    if not rows:raise ValueError('no reviewed lessons')
    torch.manual_seed(seed);torch.set_num_threads(2)
    m=load(parent_path);X=torch.stack([features(r['text']) for r in rows]);y=torch.tensor([INTENTS.index(r['intent']) for r in rows])
    opt=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=2e-4)
    for _ in range(steps):
        z,l=m(X);ce=F.cross_entropy(l,y);loss=ce
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    torch.save({'schema':'kev.semantic-breadth.public','state_dict':m.state_dict(),'parameters':m.parameter_count,'seed':seed,'intents':INTENTS},out/'semantic-breadth.pt')
    return {'steps':steps,'examples':len(rows),'parameters':m.parameter_count,'output':str(out/'semantic-breadth.pt'),'activation':'NONE'}
