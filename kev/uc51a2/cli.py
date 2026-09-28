from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from .semantic_breadth import load,predict,train,INTENTS
ROOT=Path(__file__).resolve().parents[2]
MODEL=ROOT/"models/v051a2/semantic-breadth.pt"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=["status","answer","train"])
    ap.add_argument("--message",default="");ap.add_argument("--output");ap.add_argument("--steps",type=int,default=800);ap.add_argument("--seed",type=int,default=51401)
    ap.add_argument("--parent",default=str(MODEL));ap.add_argument("--lessons")
    a=ap.parse_args()
    if a.mode=="status":
        print(json.dumps({"version":"0.51.0-alpha.2","model":str(MODEL),"sha256":sha(MODEL),"intents":list(INTENTS),
          "activation":"NONE","authority":"NONE","final_audit":"1620/2340 = 69.2%","old_v051_stress":"900/900 = 100%","claim_boundary":"research semantic classifier; not AGI/superintelligence"},indent=2))
    elif a.mode=="answer":
        if not a.message.strip():raise SystemExit("--message required")
        r=predict(load(MODEL),a.message);r.pop('embedding',None);r.update({"authority":"NONE","activation":"NONE","semantic_only":True})
        print(json.dumps(r,indent=2))
    else:
        if not a.output:raise SystemExit("--output required")
        print(json.dumps(train(a.parent,a.output,a.steps,a.seed,a.lessons),indent=2))
