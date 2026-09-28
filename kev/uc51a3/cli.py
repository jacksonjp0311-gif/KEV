from __future__ import annotations
import argparse,json
from pathlib import Path
from .alive import AliveRuntime,AliveStore,VERSION
from kev.uc51a2.semantic_breadth import INTENTS,train as train_semantic
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['status','chat','new-session','teach','export-lessons','verify-ledger','train-candidate'])
    ap.add_argument('--state-dir');ap.add_argument('--message',default='');ap.add_argument('--session');ap.add_argument('--intent');ap.add_argument('--text');ap.add_argument('--output');ap.add_argument('--steps',type=int,default=400);ap.add_argument('--seed',type=int,default=51310)
    a=ap.parse_args();store=AliveStore(a.state_dir)
    if a.mode=='status':
        s=store.read();print(json.dumps({'version':VERSION,'state_dir':str(store.dir),'active_session':s['active_session'],'facts':len(s['facts']),'typed_memory':len(s['typed_memory']),'reviewed_lessons':len(s['reviewed_lessons']),'semantic_model':s['semantic_model'],'ledger':store.verify_ledger(),'authority':'NONE'},indent=2))
    elif a.mode=='new-session':print(json.dumps({'session_id':store.new_session()},indent=2))
    elif a.mode=='chat':print(json.dumps(AliveRuntime(a.state_dir).chat(a.message,a.session),indent=2))
    elif a.mode=='teach':
        if a.intent not in INTENTS:raise SystemExit('valid --intent required: '+','.join(INTENTS))
        if not (a.text or '').strip():raise SystemExit('--text required')
        print(json.dumps(store.add_lesson(a.intent,a.text),indent=2))
    elif a.mode=='export-lessons':
        if not a.output:raise SystemExit('--output required')
        print(json.dumps(store.export_lessons(a.output),indent=2))
    elif a.mode=='verify-ledger':print(json.dumps(store.verify_ledger(),indent=2))
    else:
        if not a.output:raise SystemExit('--output required')
        lesson_path=store.dir/'reviewed-lessons.export.jsonl';store.export_lessons(lesson_path)
        parent=Path(__file__).resolve().parents[2]/'models/v051a2/semantic-breadth.pt'
        print(json.dumps(train_semantic(parent,a.output,a.steps,a.seed,lesson_path),indent=2))
