from __future__ import annotations
import json, os, re, hashlib, tempfile, time, uuid
from pathlib import Path
from urllib.request import Request, urlopen
from kev.uc51a2.semantic_breadth import load as load_semantic, predict as semantic_predict
ROOT=Path(__file__).resolve().parents[2]; DEFAULT_MODEL=ROOT/'models/v051a2/semantic-breadth.pt'; VERSION='0.51.0-alpha.3'
def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
def default_state_dir():
    if os.name=='nt': return Path(os.environ.get('LOCALAPPDATA',Path.home()/'AppData'/'Local'))/'KEV'/'alive'
    return Path.home()/'.kev'/'alive'
def _atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
def _norm_key(s): return re.sub(r'\s+',' ',s.strip().casefold()).strip(' .?!')
class AliveStore:
    def __init__(self,directory=None):
        self.dir=Path(directory or default_state_dir()).expanduser().resolve();self.state_path=self.dir/'state.json';self.ledger_path=self.dir/'ledger.jsonl';self.sessions=self.dir/'sessions';self.dir.mkdir(parents=True,exist_ok=True);self.sessions.mkdir(exist_ok=True)
        if not self.state_path.exists(): _atomic_json(self.state_path,{'schema':'kev.alive-state.v1','version':VERSION,'identity':{'name':'KEV','description':'local teachable evidence-grounded research intelligence'},'facts':{},'typed_memory':[],'reviewed_lessons':[],'active_session':None,'semantic_model':str(DEFAULT_MODEL),'created_at':now(),'updated_at':now()})
    def read(self): return json.loads(self.state_path.read_text(encoding='utf-8'))
    def write(self,s): s['updated_at']=now();_atomic_json(self.state_path,s)
    def append_event(self,kind,data):
        prev='0'*64
        if self.ledger_path.exists():
            lines=[x for x in self.ledger_path.read_text(encoding='utf-8').splitlines() if x.strip()]
            if lines:prev=json.loads(lines[-1])['hash']
        body={'schema':'kev.alive-event.v1','id':str(uuid.uuid4()),'time':now(),'kind':kind,'data':data,'prev_hash':prev};raw=json.dumps(body,sort_keys=True,separators=(',',':')).encode();body['hash']=hashlib.sha256(raw).hexdigest()
        with self.ledger_path.open('a',encoding='utf-8') as f:f.write(json.dumps(body,sort_keys=True)+'\n')
        return body
    def session_path(self,sid): return self.sessions/f'{sid}.jsonl'
    def new_session(self):
        sid=time.strftime('%Y%m%d-%H%M%S',time.localtime())+'-'+uuid.uuid4().hex[:6];s=self.read();s['active_session']=sid;self.write(s);self.append_event('SESSION_CREATED',{'session_id':sid});return sid
    def append_message(self,sid,role,content,meta=None):
        rec={'schema':'kev.alive-message.v1','time':now(),'role':role,'content':content,'meta':meta or {}}
        with self.session_path(sid).open('a',encoding='utf-8') as f:f.write(json.dumps(rec,sort_keys=True)+'\n')
        return rec
    def history(self,sid,limit=40):
        p=self.session_path(sid)
        if not p.exists():return []
        return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()][-limit:]
    def remember_fact(self,key,value,source='user',correction=False):
        k=_norm_key(key);s=self.read();old=s['facts'].get(k);rev=(old or {}).get('revision',0)+1;rec={'key':k,'value':value.strip(),'revision':rev,'source':source,'updated_at':now(),'supersedes':old['value'] if old else None};s['facts'][k]=rec;self.write(s);self.append_event('FACT_CORRECTED' if old or correction else 'FACT_ADDED',rec);return rec
    def add_typed(self,kind,text,semantic):
        s=self.read();rec={'id':str(uuid.uuid4()),'kind':kind,'text':text,'semantic':semantic,'time':now()};s['typed_memory'].append(rec);s['typed_memory']=s['typed_memory'][-500:];self.write(s);self.append_event('TYPED_MEMORY_ADDED',rec);return rec
    def add_lesson(self,intent,text):
        s=self.read();rec={'id':'lesson-'+uuid.uuid4().hex[:10],'intent':intent,'text':text,'status':'REVIEWED','time':now()};s['reviewed_lessons'].append(rec);self.write(s);self.append_event('REVIEWED_LESSON_ADDED',rec);return rec
    def export_lessons(self,path):
        s=self.read();p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('\n'.join(json.dumps({'id':x['id'],'intent':x['intent'],'text':x['text']},sort_keys=True) for x in s['reviewed_lessons'])+('\n' if s['reviewed_lessons'] else ''),encoding='utf-8');return {'path':str(p),'count':len(s['reviewed_lessons'])}
    def verify_ledger(self):
        if not self.ledger_path.exists():return {'events':0,'valid':True}
        prev='0'*64;count=0
        for line in self.ledger_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            rec=json.loads(line);h=rec.pop('hash')
            if rec['prev_hash']!=prev:return {'events':count,'valid':False,'reason':'prev_hash'}
            if hashlib.sha256(json.dumps(rec,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=h:return {'events':count,'valid':False,'reason':'hash'}
            prev=h;count+=1
        return {'events':count,'valid':True,'head':prev}
class AliveRuntime:
    def __init__(self,state_dir=None,model_path=None):
        self.store=AliveStore(state_dir);s=self.store.read();self.model_path=Path(model_path or s.get('semantic_model') or DEFAULT_MODEL);self.model=load_semantic(self.model_path)
    def _semantic(self,text):
        p=semantic_predict(self.model,text);p.pop('embedding',None);p['score_status']='UNCALIBRATED_RESEARCH_SCORE';return p
    def _surface_config(self):
        p=self.store.dir/'language-surface.json'
        if not p.exists():return {'provider':'none'}
        try:return json.loads(p.read_text(encoding='utf-8'))
        except Exception:return {'provider':'none'}
    def chat(self,message,session_id=None):
        message=str(message).strip()
        if not message:raise ValueError('message required')
        s=self.store.read();sid=session_id or s.get('active_session') or self.store.new_session();self.store.append_message(sid,'user',message);sem=self._semantic(message);lo=message.casefold();route='SEMANTIC';meta={'semantic':sem,'model':str(self.model_path)}
        if re.search(r'\bwho are you\b|\bwhat are you\b',lo):
            route='IDENTITY_QUERY';text='I am KEV, a local teachable research intelligence. I preserve typed memory and reviewed lessons; my semantic model is still experimental.'
        elif re.search(r'\bwhat do you remember\b|\bmemory status\b',lo):
            route='MEMORY_STATUS';st=self.store.read();text=f"I currently hold {len(st['facts'])} current facts, {len(st['typed_memory'])} typed memories, and {len(st['reviewed_lessons'])} reviewed lessons."
        else:
            m=re.match(r'\s*remember that\s+(.+?)\s+is\s+(.+?)[.!]?\s*$',message,re.I);corr=re.match(r'\s*(?:correction|update)\s*:\s*(.+?)\s+is\s+(.+?)[.!]?\s*$',message,re.I)
            if m or corr:
                route='FACT_CORRECTION' if corr else 'FACT_WRITE';mm=m or corr;rec=self.store.remember_fact(mm.group(1),mm.group(2),correction=bool(corr));text=f"Stored: {rec['key']} = {rec['value']} (revision {rec['revision']}).";meta['memory_write']=rec
            else:
                q=re.match(r"\s*(?:what is|what's|tell me)\s+(.+?)[?]?\s*$",message,re.I);fact=self.store.read()['facts'].get(_norm_key(q.group(1))) if q else None
                if fact:route='FACT_READ';text=fact['value'];meta['memory_read']=fact
                elif sem['intent'] in ('GOAL','CONSTRAINT','OBSERVATION','PREDICTION'):
                    route='TYPED_MEMORY_WRITE';rec=self.store.add_typed(sem['intent'],message,sem);verb=sem['intent'].casefold();text=f"I classified that as a {verb} and preserved it in typed memory.";meta['typed_memory_write']=rec
                else:text=f"I map this to {sem['intent']}. My own semantic weights are not yet broad enough for a reliable open-ended answer here."
        meta['route']=route;self.store.append_message(sid,'assistant',text,meta);self.store.append_event('CHAT_TURN',{'session_id':sid,'route':route,'semantic_intent':sem['intent']})
        return {'schema':'kev.alive-turn.v1','version':VERSION,'session_id':sid,'response':text,'route':route,'semantic':sem,'meta':meta,'authority':'NONE','tool_execution':'NONE'}
