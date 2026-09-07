import argparse
import fcntl
import json
import secrets
import signal
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from . import config, downloads
from .bench import Bench, WORKLOADS
from .discovery import CATALOG, engines, hardware, probe
from .engine import Cancelled, Engine
from .settings import Settings, capabilities, launch_args
from .store import Store
from .defaults import starting_defaults
from .recommendations import promotion_provenance, saved_qualifications
from .launch import installed_models, choose_launch


class App:
    def __init__(self):
        self.store = Store()
        self.engine = Engine()
        self.bench = Bench(self.engine,self.store)
        self.token = secrets.token_urlsafe(32)
        self.start_requests = {}

    def default_key(self,s):
        return str(Path(s.model).expanduser().resolve()) + '|' + s.backend

    def action(self,path,data):
        if path == '/api/files':
            requested = data.get('directory')
            directory = Path(requested).expanduser() if requested else config.MODELS_DIR
            if not requested and not directory.is_dir():
                directory = Path.home()
            directory = directory.resolve(strict=True)
            if not directory.is_dir():
                raise ValueError('Choose a directory.')
            entries = []
            for child in directory.iterdir():
                try:
                    is_dir = child.is_dir()
                    if is_dir or (child.is_file() and child.suffix.lower() == '.gguf'):
                        entries.append(dict(name=child.name, path=str(child), directory=is_dir,
                                            size=None if is_dir else child.stat().st_size))
                except OSError:
                    continue
            entries.sort(key=lambda e: (not e['directory'], e['name'].casefold()))
            return dict(directory=str(directory), parent=str(directory.parent), entries=entries,
                        home=str(Path.home()), models=str(config.MODELS_DIR))
        if path == '/api/discover':
            return dict(models=installed_models(),engines=engines(),catalog=CATALOG)
        if path == '/api/launch/select':
            return choose_launch(data.get('model', ''), data.get('engine', ''),
                                 data.get('backend', ''), data.get('device', ''))
        if path == '/api/launch/check':
            s = Settings.parse(data['settings'])
            error = None
            try:
                if not hardware()['gpus']:
                    raise ValueError('No NVIDIA GPU detected. Check GPU availability before starting.')
                launch_args(s, config.ENGINE_PORT)
            except (OSError, ValueError) as e:
                error = str(e)
            return dict(valid=error is None, error=error,
                        saved_exists=self.store.get('default', self.default_key(s)) is not None)
        if path == '/api/result/preview':
            r = self.store.get('result', data['result_id'])
            if not r or r['status'] != 'complete' or not r['samples']:
                raise ValueError('Choose a completed result with samples.')
            if r.get('measurement_mode') == 'warm-conversation' or r.get('quality_status') == 'failed':
                raise ValueError('This result cannot be used as a general measured configuration.')
            s = Settings.parse(r['settings'])
            if data.get('use_context'):
                if not r.get('recommended_context'):
                    raise ValueError('This result has no successful context probe.')
                s.context = r['recommended_context'] * s.slots
            return dict(settings=s.dict(), source='Experiment result · for next launch',
                        notes=['Review before starting or saving. Saved preferences are unchanged.'],
                        evidence=promotion_provenance(r, s, data.get('use_context')),
                        result_id=r['id'], use_context=bool(data.get('use_context')))
        if path == '/api/capabilities':
            s = Settings.parse(data['settings'])
            return dict(features=capabilities(s),engine={k:v for k,v in probe(s.engine).items() if k != 'help'})
        if path == '/api/default/resolve':
            s = Settings.parse(data['settings'])
            source = data.get('source','auto')
            if source not in ['auto','saved','built-in']:
                raise ValueError('Unknown defaults source.')
            saved = self.store.get('default',self.default_key(s))
            if saved and source != 'built-in':
                resolved = Settings.parse(saved)
                resolved.engine, resolved.device = s.engine, s.device
                provenance = self.store.get('default-evidence', self.default_key(s))
                if not provenance or provenance.get('settings') != saved:
                    return dict(settings=resolved.dict(), source='Saved defaults · origin unknown',
                                notes=['Legacy saved settings preserved; no measurement provenance was recorded.'])
                notes = ['Saved preferences take precedence over built-in recommendations.']
                if provenance['kind'] == 'benchmark':
                    notes.append(provenance['note'])
                    notes.extend(saved_qualifications(provenance, resolved))
                    if provenance['context']['used_headroom_estimate']:
                        notes.append('Saved context uses a headroom estimate, not an observed successful allocation.')
                return dict(settings=resolved.dict(), source='Saved defaults · ' +
                            ('historical benchmark evidence · qualified' if provenance['kind'] == 'benchmark' else 'manual preferences'),
                            notes=notes, evidence=provenance if provenance['kind'] == 'benchmark' else None)
            if source == 'saved':
                raise ValueError('No saved settings for this model and backend yet. Load a completed experiment into Launch or edit the draft, then choose “Save my settings”.')
            return starting_defaults(s)
        if path == '/api/default/load':
            s = Settings.parse(data['settings'])
            return self.store.get('default',self.default_key(s))
        if path == '/api/default/save':
            provenance = None
            if data.get('result_id'):
                r = self.store.get('result',data['result_id'])
                if not r or r['status'] != 'complete' or not r['samples']:
                    raise ValueError('Only a completed measured configuration can be promoted.')
                if r.get('measurement_mode') == 'warm-conversation':
                    raise ValueError('Warm conversation results cannot be promoted as a general cold baseline. Save launch preferences manually if desired.')
                if r.get('quality_status') == 'failed':
                    raise ValueError('Source adherence failed; this result cannot be promoted as a measured baseline.')
                s = Settings.parse(r['settings'])
                if data.get('use_context'):
                    if not r['recommended_context']:
                        raise ValueError('This run has no successful context probe.')
                    s.context = r['recommended_context'] * s.slots
                provenance = promotion_provenance(r, s, data.get('use_context'))
            else:
                s = Settings.parse(data['settings'])
            launch_args(s,config.ENGINE_PORT)
            self.store.put('default',self.default_key(s),s.dict())
            self.store.put('default-evidence', self.default_key(s),
                           provenance or dict(kind='manual', settings=s.dict()))
            return s.dict()
        if path == '/api/benchmark':
            return self.bench.submit(data)
        if path in ['/api/cancel','/api/stop']:
            with self.bench.lock:
                self.bench.cancel.set()
                self.engine.stop()
                if not self.bench.active:
                    self.bench.progress = dict(status='stopped', kind='launch')
            return dict(ok=True)
        if path == '/api/start':
            s = Settings.parse(data['settings'])
            request_id = data.get('request_id')
            if request_id is not None and (not isinstance(request_id, str) or not 1 <= len(request_id) <= 128):
                raise ValueError('Invalid start request identity.')
            launch_args(s,config.ENGINE_PORT)
            with self.bench.lock:
                if request_id in self.start_requests:
                    if self.start_requests[request_id] != s.dict():
                        raise ValueError('Start request identity already used for different settings.')
                    return dict(ok=True)
                if self.bench.active:
                    raise ValueError('Wait for the current operation or cancel it first.')
                running = self.engine.state()
                if running.get('ready') and running['settings'] == s.dict():
                    return dict(ok=True)
                if running['running']:
                    if data.get('replace_running') is not True or data.get('expected_pid') != running['pid']:
                        raise ValueError('A model is running or has changed. Refresh status and explicitly switch or restart it.')
                self.bench.active = True
                self.bench.cancel.clear()
                if request_id:
                    self.start_requests[request_id] = s.dict()
                    if len(self.start_requests) > 128:
                        del self.start_requests[next(iter(self.start_requests))]
                self.bench.progress = dict(kind='launch', status='starting', phase='Loading model',
                                           started_at=time.time(), settings=s.dict(), request_id=request_id)
            def start():
                try:
                    self.engine.start(s,self.bench.cancel)
                    self.bench.update(status='serving',phase='Ready')
                except Cancelled:
                    self.bench.update(status='cancelled')
                except Exception as e:
                    self.bench.update(status='failed',error=str(e))
                finally:
                    with self.bench.lock:
                        self.bench.active = False
            threading.Thread(target=start,daemon=True).start()
            return dict(ok=True)
        if path == '/api/download':
            entry = next((e for e in CATALOG if e['id']==data['id']),None)
            if entry is None:
                raise ValueError('Unknown catalogue model.')
            return downloads.start(entry).as_dict()
        if path == '/api/download/cancel':
            return dict(cancelled=downloads.cancel(data['id']))
        raise ValueError('Unknown action')


def serve(host='127.0.0.1', port=8082):
    config.STATE_DIR.mkdir(parents=True,exist_ok=True)
    lock = (config.STATE_DIR/'panel.lock').open('w')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError('Another lllm2 panel owns this state directory.') from error
    app = App()
    hostnames = {'127.0.0.1','localhost',socket.gethostname().lower(),socket.getfqdn().lower()}
    allowed_hosts = {f'{name}:{port}' for name in hostnames}

    class Handler(BaseHTTPRequestHandler):
        def send(self,status,body,kind='application/json'):
            raw = json.dumps(body).encode() if kind == 'application/json' else body
            self.send_response(status)
            self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError):
                pass

        def valid_host(self):
            # The accepted socket identifies the local interface used by this
            # request, including LAN addresses on multi-interface workstations.
            local_host = f'{self.connection.getsockname()[0]}:{port}'
            if self.headers.get('Host','').lower() not in allowed_hosts | {local_host}:
                self.send(403,dict(error='Use this workstation’s panel address or hostname.'))
                return False
            return True

        def do_GET(self):
            if not self.valid_host():
                return
            path = urlparse(self.path).path
            if path == '/':
                self.send(200,Path(__file__).with_name('static').joinpath('index.html').read_bytes(),'text/html; charset=utf-8')
            elif path == '/static/panel.js':
                self.send(200,Path(__file__).with_name('static').joinpath('panel.js').read_bytes(),'text/javascript; charset=utf-8')
            elif path == '/api/status':
                self.send(200,dict(token=app.token,engine=app.engine.state(),job=app.bench.snapshot(),hardware=hardware(),
                                   downloads=downloads.all_downloads(),endpoint=app.engine.base+'/v1',
                                   paths=dict(models=str(config.MODELS_DIR),engines=[str(p) for p in config.ENGINE_ROOTS]),workloads=WORKLOADS))
            elif path in ['/api/results','/api/results/export']:
                rows = app.store.list('summary' if path == '/api/results' else 'result')
                self.send(200,rows)
            else:
                self.send(404,dict(error='Not found'))

        def do_POST(self):
            if not self.valid_host():
                return
            origin = self.headers.get('Origin')
            if self.headers.get('X-LLLM2-Token') != app.token or (origin and origin.lower() != f'http://{self.headers.get("Host","").lower()}'):
                self.send(403,dict(error='Reload the panel to renew its session.'))
                return
            try:
                size = int(self.headers.get('Content-Length','0'))
                if not 0<size<=2_000_000:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(size))
                self.send(200,app.action(urlparse(self.path).path,data))
            except (ValueError,KeyError,TypeError,OSError) as e:
                self.send(400,dict(error=str(e)))
            except Exception as e:
                app.engine.log('Panel error: ' + str(e))
                self.send(500,dict(error=str(e)))

        def log_message(self,*args):
            pass

    server = ThreadingHTTPServer((host,port),Handler)
    def shutdown(*_):
        threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,shutdown)
    signal.signal(signal.SIGINT,shutdown)
    print(f'lllm2: http://127.0.0.1:{port}',flush=True)
    if host != '127.0.0.1':
        print(f'LAN panel: http://{socket.gethostname() if host == "0.0.0.0" else host}:{port} (listening on {host})',flush=True)
        print('LAN access has no login or TLS: anyone who can reach this port can control the workbench. Use only on a trusted network.',flush=True)
    try:
        server.serve_forever()
    finally:
        app.bench.cancel.set()
        app.engine.stop()
        server.server_close()
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description='lllm2 local LLM workbench')
    parser.add_argument('--port',type=int,default=8082)
    parser.add_argument('--host',default='127.0.0.1',help='IPv4 bind address (default: localhost; use 0.0.0.0 for trusted-LAN access without authentication or TLS)')
    args = parser.parse_args(argv)
    serve(args.host, args.port)
