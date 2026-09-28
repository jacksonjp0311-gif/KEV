from __future__ import annotations
import argparse,json,threading,webbrowser
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from .alive import AliveRuntime,AliveStore
ROOT=Path(__file__).resolve().parents[2]; WEB=ROOT/'desktop/v051a3'
def make_server(state_dir,port=39061):
    runtime=AliveRuntime(state_dir);store=runtime.store
    class H(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def sendj(self,obj,code=200):
            b=json.dumps(obj).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
        def do_GET(self):
            if self.path=='/api/status':
                s=store.read();return self.sendj({'version':s['version'],'active_session':s['active_session'],'facts':len(s['facts']),'typed_memory':len(s['typed_memory']),'reviewed_lessons':len(s['reviewed_lessons']),'ledger':store.verify_ledger()})
            if self.path.startswith('/api/history'):
                sid=store.read().get('active_session');return self.sendj({'session_id':sid,'messages':store.history(sid,80) if sid else []})
            p=WEB/('index.html' if self.path in ('/','/index.html') else self.path.lstrip('/'))
            if p.exists() and p.is_file():
                b=p.read_bytes();ct='text/html' if p.suffix=='.html' else ('text/javascript' if p.suffix=='.js' else 'text/css');self.send_response(200);self.send_header('Content-Type',ct);self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b);return
            self.send_error(404)
        def do_POST(self):
            try:
                n=int(self.headers.get('Content-Length','0'));body=json.loads(self.rfile.read(n) or b'{}')
                if self.path=='/api/chat':return self.sendj(runtime.chat(body.get('message',''),body.get('session_id')))
                if self.path=='/api/new-session':return self.sendj({'session_id':store.new_session()})
                if self.path=='/api/teach':return self.sendj(store.add_lesson(body.get('intent',''),body.get('text','')))
                self.send_error(404)
            except Exception as e:self.sendj({'error':type(e).__name__,'message':str(e)},400)
    return ThreadingHTTPServer(('127.0.0.1',port),H)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--state-dir');ap.add_argument('--port',type=int,default=39061);ap.add_argument('--no-browser',action='store_true');a=ap.parse_args();srv=make_server(a.state_dir,a.port);url=f'http://127.0.0.1:{srv.server_port}/';print(url,flush=True)
    if not a.no_browser:threading.Timer(.5,lambda:webbrowser.open(url)).start()
    try:srv.serve_forever()
    except KeyboardInterrupt:pass
    finally:srv.server_close()
