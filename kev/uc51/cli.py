from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from .semantic_core import load,predict,canon,execute,INTENTS,train
ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'models/v051/semantic-core.pt'

def answer(message,model=DEFAULT):
    m=load(model); p=predict(m,message); ans=execute(p['intent'],message)
    emb=bytes(int(max(0,min(255,(v+1)*127.5))) for v in p.pop('embedding'))
    return {'schema':'kev.semantic-frame.v051','intent':p['intent'],'score':p['score'],'bounded_answer':ans,
            'canonical_tokens':canon(message),'semantic_fingerprint':hashlib.sha256(emb).hexdigest(),
            'authority':'NONE','activation':'NONE','claim_boundary':'Bounded semantic relation classifier; exact evidence remains governed by KEV grounding.'}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['status','answer','train']);ap.add_argument('--message',default='');ap.add_argument('--model',default=str(DEFAULT));ap.add_argument('--output');ap.add_argument('--steps',type=int,default=900);ap.add_argument('--seed',type=int,default=51021);a=ap.parse_args()
    if a.mode=='train':
        if not a.output: raise SystemExit('--output is required')
        print(json.dumps(train(a.output,a.steps,a.seed),indent=2))
    elif a.mode=='status':
        ck=__import__('torch').load(a.model,map_location='cpu',weights_only=True)
        print(json.dumps({'version':'0.51.0-alpha.1','model':a.model,'parameters':ck['parameters'],'intents':list(INTENTS),'activation':'NONE'},indent=2))
    else: print(json.dumps(answer(a.message,a.model),indent=2))
