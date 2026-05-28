import os
import zipfile

def export_to_kaggle():
    target_zip = "lnn_kaggle.zip"
    
    # Files and extensions we actually want to include
    include_extensions = ['.py', '.yaml', '.ipynb']
    exclude_dirs = ['checkpoints', '__pycache__', '.git', 'godot']

    print(f"Creating {target_zip}...")
    
    with zipfile.ZipFile(target_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk('.'):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if any(file.endswith(ext) for ext in include_extensions) or file == "requirements.txt":
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, '.')
                    zf.write(file_path, arcname)
                    print(f"Added: {arcname}")
                    
    print(f"\nDone! You can now upload '{target_zip}' to Kaggle.")

if __name__ == "__main__":
    export_to_kaggle()
