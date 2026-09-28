from __future__ import annotations
import hashlib,json,random,re
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

INTENTS=('COPY_VALUE','SUPERSEDES','MAGNITUDE','NEGATE','REFERENCE','ACTIVE_SELECTION','RUN_STATUS','RECEIPT_VALUE','EVIDENCE_CONSISTENCY')
D=1024; EMB=80
CANON={
'current':'CURRENT','now':'CURRENT','latest':'CURRENT','presently':'CURRENT','governs':'CURRENT','applies':'CURRENT',
'old':'OLD','earlier':'OLD','previous':'OLD','prior':'OLD','former':'OLD','formerly':'OLD','obsolete':'OLD','retired':'OLD','retires':'OLD','discard':'OLD',
'replace':'UPDATE','replaced':'UPDATE','replacement':'UPDATE','replaces':'UPDATE','supersede':'UPDATE','superseded':'UPDATE','superseding':'UPDATE','correction':'UPDATE','renamed':'UPDATE','update':'UPDATE','updated':'UPDATE','revision':'UPDATE','establishes':'CURRENT','established':'CURRENT','authoritative':'CURRENT','force':'CURRENT','valid':'CURRENT',
'identifier':'IDENTIFIER','designation':'IDENTIFIER','label':'IDENTIFIER','code':'IDENTIFIER','tag':'IDENTIFIER','name':'IDENTIFIER',
'absolute':'MAGNITUDE','magnitude':'MAGNITUDE','unsigned':'MAGNITUDE','nonnegative':'MAGNITUDE','distance':'MAGNITUDE','size':'MAGNITUDE',
'negate':'NEGATE','negation':'NEGATE','inverse':'NEGATE','opposite':'NEGATE','sign-reversed':'NEGATE','sign':'NEGATE',
'active':'ACTIVE','running':'ACTIVE','serving':'ACTIVE','selected':'ACTIVE','deployed':'ACTIVE','live':'ACTIVE','inactive':'INACTIVE','candidate':'INACTIVE','challenger':'INACTIVE',
'receipt':'RECEIPT','record':'RECEIPT','recorded':'RECEIPT','observed':'RECEIPT','observation':'RECEIPT','output':'RESULT','result':'RESULT','returned':'RESULT','produced':'RESULT','measured':'RECEIPT','outcome':'RESULT',
'status':'STATUS','state':'STATUS','completed':'COMPLETED','interrupted':'INTERRUPTED','failed':'FAILED','pending':'PENDING',
'bears':'RELATE','carries':'RELATE','belongs':'RELATE','attached':'RELATE','maps':'RELATE','identifies':'RELATE','consistent':'CONSISTENT','agreement':'CONSISTENT','conflict':'CONFLICT','conflicting':'CONFLICT','disagree':'CONFLICT','disagreement':'CONFLICT','unknown':'UNKNOWN','missing':'UNKNOWN','absent':'UNKNOWN'}

def _h(s): return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4],'big')%D

def canon(text):
    raw=re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*|-?\d+",text.casefold()); out=[]
    for w in raw:
        if re.fullmatch(r'-?\d+',w): out.append('NUMBER')
        else: out.append(CANON.get(w,w))
    st=set(out)
    if 'OLD' in st and 'CURRENT' in st: out.append('REL_SUPERSEDES')
    if 'ACTIVE' in st and 'INACTIVE' in st: out.append('REL_ACTIVE_SELECTION')
    if 'RECEIPT' in st and 'RESULT' in st: out.append('REL_RECEIPT_VALUE')
    if 'MAGNITUDE' in st: out.append('REL_MAGNITUDE')
    if 'NEGATE' in st: out.append('REL_NEGATE')
    return out

def features(text):
    t=canon(text); x=torch.zeros(D)
    for w in t:x[_h('u:'+w)]+=1
    for a,b in zip(t,t[1:]):x[_h('b:'+a+'_'+b)]+=1
    return torch.log1p(x)

class SemanticCore(nn.Module):
    def __init__(self):
        super().__init__(); self.enc=nn.Sequential(nn.Linear(D,256),nn.GELU(),nn.LayerNorm(256),nn.Linear(256,EMB)); self.head=nn.Linear(EMB,len(INTENTS))
    def forward(self,x):
        z=F.normalize(self.enc(x),dim=-1); return z,self.head(z)
    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())

def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=True); m=SemanticCore();m.load_state_dict(ck['state_dict']);m.eval();return m

def predict(m,text):
    with torch.no_grad(): z,l=m(features(text)[None,:]); p=torch.softmax(l,1)[0]; i=int(p.argmax())
    return {'intent':INTENTS[i],'score':float(p[i]),'embedding':z[0].tolist()}

def execute(intent,text):
    lo=text.casefold(); vals=re.findall(r'\b(?:amber|blue|copper|green|ivory|orange|red|silver|violet|white)\b',lo); nums=[int(x) for x in re.findall(r'(?<![a-z0-9])-?\d+(?![a-z0-9])',lo)]
    if intent=='COPY_VALUE': return vals[0] if vals else None
    if intent=='SUPERSEDES': return vals[-1] if vals else None
    if intent=='MAGNITUDE': return str(abs(nums[-1])) if nums else None
    if intent=='NEGATE': return str(-nums[-1]) if nums else None
    if intent=='ACTIVE_SELECTION': return vals[-1] if vals else None
    if intent=='EVIDENCE_CONSISTENCY': return 'consistent' if len(vals)>=2 and vals[0]==vals[1] else 'conflict'
    return None

def train(out_dir,steps=800,seed=51001):
    raise RuntimeError("Public source snapshot omits generated training corpus/checkpoints. See docs and MODEL_WEIGHTS.md for the reproducible research package.")
