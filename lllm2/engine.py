import collections
import csv
import shlex
from pathlib import Path
import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.request
import urllib.error
from . import config
from .discovery import command, hardware, engine_environment
from .settings import launch_args


class Cancelled(Exception):
    pass


class GPUUnavailable(RuntimeError):
    pass


class ResourceConflict(GPUUnavailable):
    pass


def gpu_processes(output):
    """Separate normal browser GPU helpers from competing compute workloads.

    nvidia-smi can classify Chrome's Wayland GPU helper as compute. Prefer
    /proc identity, since some drivers report only the executable, not argv.
    This is a coexistence check, not a security boundary or memory guarantee.
    """
    desktop, competing = [], []
    browsers = {'chrome', 'chromium', 'chromium-browser', 'google-chrome',
                'google-chrome-stable', 'brave', 'brave-browser', 'msedge'}
    for row in csv.reader(output.splitlines()):
        if not row:
            continue
        if len(row) < 2 or not row[0].strip().isdigit():
            competing.append('Unrecognised GPU process record')
            continue
        pid, reported = row[0].strip(), ','.join(row[1:]).strip()
        try:
            argv = shlex.split(reported)
        except ValueError:
            argv = [reported]
        executable = argv[0] if argv else reported
        try:
            proc = Path('/proc') / pid
            executable = str((proc / 'exe').readlink())
            argv = (proc / 'cmdline').read_bytes().decode('utf-8', 'replace').split('\0')
        except OSError:
            pass
        name = Path(executable).name
        summary = f'{pid}, {name}'
        if name in browsers and '--type=gpu-process' in argv:
            desktop.append(summary)
        else:
            competing.append(summary)
    return desktop, competing


class Engine:
    def __init__(self):
        self.process = None
        self.settings = None
        self.argv = []
        self.lines = collections.deque(maxlen=1000)
        self.log_lock = threading.Lock()
        self.guard = threading.RLock()
        self.base = f'http://127.0.0.1:{config.ENGINE_PORT}'

    def log(self, message):
        with self.log_lock:
            self.lines.append(message)

    def state(self):
        with self.log_lock:
            logs = list(self.lines)[-200:]
        return dict(running=self.process is not None and self.process.poll() is None,
                    pid=self.process.pid if self.process else None,
                    settings=self.settings.dict() if self.settings else None, argv=self.argv, logs=logs)

    def request(self, path, body=None, timeout=30):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read(4000).decode('utf-8','replace')
            raise RuntimeError(f'{path}: HTTP {e.code}: {detail}') from e

    def stop(self):
        with self.guard:
            p = self.process
            if p is None:
                return
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired as e:
                        raise GPUUnavailable('Owned engine did not exit after kill; queue halted. Inspect workstation GPU.') from e
            self.process = None
            self.settings = None

    def _logs(self, p):
        for line in p.stdout:
            self.log(line.rstrip())
        p.stdout.close()

    def start(self, s, cancel, timeout=180):
        argv = launch_args(s, config.ENGINE_PORT)
        self.stop()
        if cancel.is_set():
            raise Cancelled()
        hw = hardware()
        if not hw['gpus']:
            raise GPUUnavailable('NVIDIA GPU unavailable: ' + str(hw['error']))
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('127.0.0.1', config.ENGINE_PORT))
            except OSError as e:
                raise ResourceConflict(f'Port {config.ENGINE_PORT} is occupied by another process. Stop it yourself or change LLLM2_ENGINE_PORT.') from e
        rc, processes = command(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],4)
        if rc != 0:
            raise GPUUnavailable('Cannot check GPU ownership: ' + processes[:300])
        desktop, competing = gpu_processes(processes)
        if competing:
            raise ResourceConflict('Other GPU compute processes detected; stop the old model/server first: ' + '; '.join(competing)[:500])
        if desktop:
            self.log('Allowing desktop browser GPU processes: ' + '; '.join(desktop) + '. Their VRAM and activity remain part of this workstation benchmark.')
        with self.guard:
            if cancel.is_set():
                raise Cancelled()
            self.argv = argv
            self.settings = s
            self.log('Launching: ' + ' '.join(argv))
            p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, errors='replace', start_new_session=True,
                                 env=engine_environment(argv[0]))
            self.process = p
            threading.Thread(target=self._logs,args=(p,),daemon=True).start()
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                if cancel.wait(.25):
                    raise Cancelled()
                if p.poll() is not None:
                    raise RuntimeError('Engine exited during load. See engine log.')
                try:
                    self.request('/health',timeout=1)
                    break
                except Exception:
                    pass
            else:
                raise TimeoutError('Engine startup exceeded timeout.')
            body = dict(messages=[dict(role='user',content='Hello')])
            if s.effort != 'default':
                body['reasoning_effort'] = s.effort
            self.guarded_request('/apply-template',body,cancel,min(timeout,30))
        except BaseException:
            self.stop()
            raise

    def guarded_request(self, path, body, cancel, timeout):
        # A wall-clock bound, even if the peer dribbles bytes indefinitely.
        result = []
        error = []
        done = threading.Event()
        def work():
            try:
                result.append(self.request(path,body,timeout))
            except Exception as e:
                error.append(e)
            finally:
                done.set()
        threading.Thread(target=work,daemon=True).start()
        deadline = time.monotonic() + timeout
        while not done.wait(.1):
            if cancel.is_set():
                self.stop()
                raise Cancelled()
            if time.monotonic() > deadline:
                self.stop()
                raise TimeoutError(f'{path} exceeded {timeout}s wall-clock timeout.')
        if cancel.is_set():
            raise Cancelled()
        if error:
            raise error[0]
        return result[0]
