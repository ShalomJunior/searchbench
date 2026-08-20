import os
import time
import subprocess

def get_mtimes(directory):
    mtimes = {}
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                mtimes[path] = os.path.getmtime(path)
    return mtimes

def watch_and_format(directories):
    print(f"Watching {directories} for changes to format with Ruff...")
    last_mtimes = get_mtimes(directories[0])
    for d in directories[1:]:
        last_mtimes.update(get_mtimes(d))
        
    ruff_path = "/home/masterjs/kaggle/.venv/bin/ruff"
    
    while True:
        time.sleep(1)
        current_mtimes = {}
        for d in directories:
            current_mtimes.update(get_mtimes(d))
        
        changed_files = []
        for path, mtime in current_mtimes.items():
            # If the file is new or modified
            if path not in last_mtimes or mtime > last_mtimes[path]:
                changed_files.append(path)
                
        if changed_files:
            try:
                # 1. Fix imports and lint errors
                subprocess.run([ruff_path, "check", "--fix", "-q"] + changed_files)
                # 2. Format the code
                subprocess.run([ruff_path, "format", "-q"] + changed_files)
                print(f"Ruff formatted: {', '.join(changed_files)}")
            except Exception as e:
                print(f"Error formatting: {e}")
                
            # Update mtimes after formatting so we don't infinitely format
            last_mtimes = {}
            for d in directories:
                last_mtimes.update(get_mtimes(d))

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(project_root, 'src')
    exp_dir = os.path.join(project_root, 'experiments')
    watch_and_format([src_dir, exp_dir])
