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

def watch_and_format(directory):
    print(f"Watching {directory} for changes to format with Black...")
    last_mtimes = get_mtimes(directory)
    black_path = "/home/masterjs/kaggle/.venv/bin/black"
    
    while True:
        time.sleep(1)
        current_mtimes = get_mtimes(directory)
        
        changed_files = []
        for path, mtime in current_mtimes.items():
            # If the file is new or modified
            if path not in last_mtimes or mtime > last_mtimes[path]:
                changed_files.append(path)
                
        if changed_files:
            try:
                subprocess.run([black_path, "-q"] + changed_files)
                print(f"Formatted: {', '.join(changed_files)}")
            except Exception as e:
                print(f"Error formatting: {e}")
                
            # Update mtimes after formatting so we don't infinitely format
            last_mtimes = get_mtimes(directory)

if __name__ == "__main__":
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
    watch_and_format(src_dir)
