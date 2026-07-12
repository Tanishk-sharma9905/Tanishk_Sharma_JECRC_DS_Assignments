import subprocess
import os
import tempfile
import sys
import shutil
import time
import datetime
import logging

class Sandbox:
    def __init__(self, project_name="default_project", base_dir="workspace"):
        import re
        # Sanitize project_name to prevent path traversal
        project_name = re.sub(r'[^A-Za-z0-9_-]', '', project_name.strip())
        if not project_name:
            project_name = "default_project"
            
        abs_base = os.path.abspath(base_dir)
        self.project_dir = os.path.abspath(os.path.join(abs_base, project_name))
        
        # Verify resolved path remains under base_dir to prevent traversal
        if not self.project_dir.startswith(abs_base):
            raise ValueError("Path traversal attempt detected in Sandbox workspace setup!")
            
        self.uploads_dir = os.path.join(self.project_dir, "uploads")
        self.outputs_dir = os.path.join(self.project_dir, "outputs")
        self.temp_dir = os.path.join(self.project_dir, "temp")
        
        os.makedirs(self.uploads_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Setup logging for Sandbox
        self.logger = logging.getLogger(f"Sandbox-{project_name}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            self.logger.addHandler(ch)
    
    def _clean_dir(self, directory):
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            try:
                if os.path.isfile(filepath) or os.path.islink(filepath):
                    os.unlink(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
            except Exception as e:
                # Task 7: Sandbox Error Handling (Do not silently ignore)
                self.logger.warning(f"Failed to clean {filepath}: {e}")

    def execute_code(self, code: str, timeout: int = 60) -> dict:
        """
        Executes the given Python code in a subprocess.
        Returns a dictionary with 'stdout', 'stderr', 'success',
        'files_generated', 'execution_time', and 'exit_status'.
        """
        self._clean_dir(self.temp_dir)
                
        code_file_path = os.path.join(self.temp_dir, "script.py")
        try:
            allowed_write = [self.project_dir, self.uploads_dir, self.outputs_dir, self.temp_dir]
            allowed_write.append(tempfile.gettempdir())
            allowed_write_repr = repr([os.path.abspath(p) for p in allowed_write])
            
            security_shield = f"""# Sandbox Security Shield
import sys
import os
import builtins
import socket
import subprocess
import importlib
import shutil
import multiprocessing

# Disable reload bypass
_original_reload = importlib.reload
def restricted_reload(module):
    raise PermissionError("Module reloading is disabled in the sandbox.")
importlib.reload = restricted_reload

# Disable subprocesses
def restricted_spawn(*args, **kwargs):
    raise PermissionError("Subprocess spawning is disabled in the sandbox.")

class RestrictedPopen(subprocess.Popen):
    def __init__(self, *args, **kwargs):
        raise PermissionError("Subprocess spawning is disabled in the sandbox.")

subprocess.Popen = RestrictedPopen
subprocess.run = restricted_spawn
subprocess.call = restricted_spawn
subprocess.check_call = restricted_spawn
subprocess.check_output = restricted_spawn
os.system = restricted_spawn
for name in dir(os):
    if name.startswith("exec") or name.startswith("spawn") or name == "system":
        try:
            setattr(os, name, restricted_spawn)
        except Exception:
            pass

# Disable fork and multiprocessing
def restricted_process(*a, **kw):
    raise PermissionError("Process spawning is disabled in the sandbox.")
multiprocessing.Process = restricted_process
multiprocessing.Pool = restricted_process
if hasattr(os, "fork"):
    os.fork = restricted_spawn

# Disable socket/network
class RestrictedSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        raise PermissionError("Network connections are disabled in the sandbox.")
socket.socket = RestrictedSocket

# --- Path allow-list setup (needed by both the ctypes guard below and the
# file-access restrictions further down) ---
allowed_write_prefixes = [os.path.abspath(p) for p in {allowed_write_repr}]
allowed_read_prefixes = allowed_write_prefixes + [os.path.abspath(p) for p in sys.path if p] + [os.path.abspath(sys.prefix)]
if hasattr(sys, "base_prefix"):
    allowed_read_prefixes.append(os.path.abspath(sys.base_prefix))
if hasattr(sys, "exec_prefix"):
    allowed_read_prefixes.append(os.path.abspath(sys.exec_prefix))

def is_subpath(child, parent):
    child = os.path.abspath(child)
    parent = os.path.abspath(parent)
    return child.startswith(parent + os.sep) or child == parent

def check_write_path(path):
    abs_path = os.path.abspath(str(path))
    if not any(is_subpath(abs_path, p) for p in allowed_write_prefixes):
        raise PermissionError(f"Modifying path '{{path}}' is disabled in the sandbox.")

# Outbound HTTP / web requests are blocked at the socket layer.
# Native library loading via ctypes is restricted to an allow-list rather
# than blocked outright: legitimate installed packages (numpy, scipy,
# scikit-learn, etc.) load their own bundled runtime DLLs/shared objects
# on import (e.g. vcomp140.dll on Windows), and blocking that unconditionally
# breaks those imports entirely. Loads are only permitted when they resolve
# to a path inside the interpreter's own environment (venv/site-packages/
# stdlib), or are bare library names resolved via the OS's own system
# search path (e.g. "kernel32", "libc.so.6"). Everything else is denied.
try:
    import ctypes

    def _is_allowed_native_lib(name):
        if name is None:
            # ctypes.CDLL(None) / PyDLL(None) exposes symbols already loaded
            # in the current process; it cannot load new arbitrary code.
            return True
        name_str = str(name)
        has_dir = (os.sep in name_str) or (os.altsep and os.altsep in name_str) or (
            len(name_str) > 1 and name_str[1] == ":"
        )
        if not has_dir:
            # Bare library name: resolved via the OS's own trusted system
            # search path, not an arbitrary filesystem path.
            return True
        abs_path = os.path.abspath(name_str)
        return any(is_subpath(abs_path, p) for p in allowed_read_prefixes)

    def _make_guarded_loader(original_loader, label):
        def guarded(name=None, *a, **kw):
            if not _is_allowed_native_lib(name):
                raise PermissionError(
                    f"Loading native library '{{name}}' from outside the sandboxed "
                    f"environment is disabled ({{label}})."
                )
            return original_loader(name, *a, **kw)
        return guarded

    ctypes.CDLL = _make_guarded_loader(ctypes.CDLL, "CDLL")
    ctypes.PyDLL = _make_guarded_loader(ctypes.PyDLL, "PyDLL")
    if hasattr(ctypes, "WinDLL"):
        ctypes.WinDLL = _make_guarded_loader(ctypes.WinDLL, "WinDLL")
    if hasattr(ctypes, "OleDLL"):
        ctypes.OleDLL = _make_guarded_loader(ctypes.OleDLL, "OleDLL")
    ctypes.cdll.LoadLibrary = _make_guarded_loader(ctypes.cdll.LoadLibrary, "LoadLibrary")
except Exception:
    pass

# Restrict file access
original_open = builtins.open

def restricted_open(file, mode='r', *args, **kwargs):
    try:
        # If file is descriptor, allow it
        if isinstance(file, int):
            return original_open(file, mode, *args, **kwargs)
        
        abs_file = os.path.abspath(str(file))
        is_write = any(c in mode for c in ['w', 'a', 'x', '+'])
        
        if is_write:
            if not any(is_subpath(abs_file, p) for p in allowed_write_prefixes):
                raise PermissionError(f"Writing to '{{file}}' is disabled in the sandbox.")
    except PermissionError:
        raise
    except Exception:
        # Fallback to allow if any check failed unexpectedly
        pass
    return original_open(file, mode, *args, **kwargs)

builtins.open = restricted_open

# Disable filesystem-mutating functions outside workspace
original_remove = os.remove
def restricted_remove(path, *a, **kw):
    check_write_path(path)
    return original_remove(path, *a, **kw)
os.remove = restricted_remove

original_unlink = os.unlink
def restricted_unlink(path, *a, **kw):
    check_write_path(path)
    return original_unlink(path, *a, **kw)
os.unlink = restricted_unlink

original_rename = os.rename
def restricted_rename(src, dst, *a, **kw):
    check_write_path(src)
    check_write_path(dst)
    return original_rename(src, dst, *a, **kw)
os.rename = restricted_rename

original_replace = os.replace
def restricted_replace(src, dst, *a, **kw):
    check_write_path(src)
    check_write_path(dst)
    return original_replace(src, dst, *a, **kw)
os.replace = restricted_replace

original_rmdir = os.rmdir
def restricted_rmdir(path, *a, **kw):
    check_write_path(path)
    return original_rmdir(path, *a, **kw)
os.rmdir = restricted_rmdir

original_rmtree = shutil.rmtree
def restricted_rmtree(path, *a, **kw):
    check_write_path(path)
    return original_rmtree(path, *a, **kw)
shutil.rmtree = restricted_rmtree

original_move = shutil.move
def restricted_move(src, dst, *a, **kw):
    check_write_path(src)
    check_write_path(dst)
    return original_move(src, dst, *a, **kw)
shutil.move = restricted_move

original_copy = shutil.copy
def restricted_copy(src, dst, *a, **kw):
    abs_dst = os.path.abspath(str(dst))
    if not any(is_subpath(abs_dst, p) for p in allowed_write_prefixes):
        raise PermissionError(f"Copying to '{{dst}}' is disabled in the sandbox.")
    return original_copy(src, dst, *a, **kw)
shutil.copy = restricted_copy

original_copytree = shutil.copytree
def restricted_copytree(src, dst, *a, **kw):
    abs_dst = os.path.abspath(str(dst))
    if not any(is_subpath(abs_dst, p) for p in allowed_write_prefixes):
        raise PermissionError(f"Copying to '{{dst}}' is disabled in the sandbox.")
    return original_copytree(src, dst, *a, **kw)
shutil.copytree = restricted_copytree

def denied_mutate(*a, **kw):
    raise PermissionError("chmod and chown are disabled in the sandbox.")
os.chmod = denied_mutate
if hasattr(os, "chown"):
    os.chown = denied_mutate

# Low-level os.open write restriction
original_os_open = os.open
def restricted_os_open(path, flags, mode=0o777, *a, **kw):
    # Check if flags denote write operations
    is_write = (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)) != 0
    if is_write:
        check_write_path(path)
    return original_os_open(path, flags, mode, *a, **kw)
os.open = restricted_os_open

# End of Sandbox Security Shield
import matplotlib
matplotlib.use('Agg')
"""
            with open(code_file_path, "w", encoding="utf-8") as f:
                f.write(security_shield)
                f.write("\n")
                f.write(code)
        except Exception as e:
            self.logger.error(f"Failed to write script.py: {e}")
            return {"stdout": "", "stderr": f"Sandbox Error: {e}", "success": False, "files_generated": [], "execution_time": 0, "exit_status": "IO_ERROR", "warnings": []}

        result = {
            "stdout": "",
            "stderr": "",
            "success": False,
            "files_generated": [],
            "execution_time": 0.0,
            "exit_status": None,
            "warnings": []
        }

        start_time = time.time()
        
        # Task 8: Strengthen Sandbox
        try:
            # joblib/loky (used internally by scikit-learn) tries to shell
            # out to OS tools (e.g. `wmic` on Windows) to count physical
            # CPU cores. Subprocess spawning is disabled in the sandbox, so
            # that probe always fails; joblib catches it and falls back to
            # the logical core count, but it emits a noisy warning (with an
            # embedded traceback) every time. Pre-setting LOKY_MAX_CPU_COUNT
            # tells loky the core count up front so it never attempts the
            # subprocess call at all, eliminating the warning at the source.
            child_env = os.environ.copy()
            child_env.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

            process = subprocess.Popen(
                [sys.executable, "script.py"],
                cwd=self.temp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=child_env
            )
            stdout, stderr = process.communicate(timeout=timeout)
            
            result["stdout"] = stdout[:50000] # Limit output size to prevent memory bloat
            result["stderr"] = stderr[:50000]
            result["success"] = process.returncode == 0
            result["exit_status"] = process.returncode
            
            import re
            warning_pattern = re.compile(r'.*:\d+: \w*Warning: .*')
            warnings_found = [line for line in stderr.splitlines() if warning_pattern.match(line)]
            if warnings_found:
                result["warnings"].extend(warnings_found)
            
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            result["stderr"] = f"Execution timed out after {timeout} seconds.\n" + (stderr or "")
            result["success"] = False
            result["exit_status"] = "TIMEOUT"
            self.logger.warning("Sandbox execution timed out.")
        except Exception as e:
            result["stderr"] = f"Unexpected execution error: {str(e)}"
            result["success"] = False
            result["exit_status"] = "EXCEPTION"
            self.logger.error(f"Sandbox execution failed: {e}")

        result["execution_time"] = round(time.time() - start_time, 2)
        
        log_path = os.path.join(self.temp_dir, "execution.log")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"Timestamp: {datetime.datetime.now()}\n")
                f.write(f"Success: {result['success']}\n")
                f.write(f"Execution Time: {result['execution_time']}s\n")
                f.write(f"Exit Status: {result['exit_status']}\n")
        except Exception as e:
            self.logger.warning(f"Failed to write execution log: {e}")

        # Version outputs inside outputs_dir
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_outputs_dir = os.path.join(self.outputs_dir, run_id)
        os.makedirs(run_outputs_dir, exist_ok=True)
        
        for filename in os.listdir(self.temp_dir):
            if filename not in ["script.py", "execution.log"]:
                src = os.path.join(self.temp_dir, filename)
                dst = os.path.join(run_outputs_dir, filename)
                try:
                    shutil.move(src, dst)
                    result["files_generated"].append(dst)
                except Exception as e:
                    self.logger.warning(f"Failed to move generated file {filename}: {e}")
                
        return result

