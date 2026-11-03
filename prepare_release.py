"""Build an explicit public-file manifest; omit environments and legacy claims."""
import base64
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
paths = [p for folder in ['learnlens', 'tests', 'docs', '.github', '.streamlit']
         for p in (root/folder).rglob('*')
         if p.is_file() and '__pycache__' not in p.parts]
names = ['README.md','RESEARCH_AUDIT.md','RESEARCH_PROTOCOL.md','VALIDATION.md',
    'LICENSE','.gitignore','.dockerignore','Dockerfile','app.py','run_benchmark.py',
    'export_site.py','execute_notebook.py','prepare_release.py',
    'requirements.txt','requirements-dev.txt','requirements-lock.txt',
    'LearnLens_Colab.ipynb','outputs/benchmark.json','outputs/benchmark.png',
    'outputs/LearnLens_executed.ipynb','outputs/notebook_benchmark.json',
    'outputs/notebook_benchmark.png']
paths += [root/name for name in names]
release = root/'release'
release.mkdir(exist_ok=True)
entries = []
with zipfile.ZipFile(release/'learnlens.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(set(paths)):
        if not p.is_file():
            raise FileNotFoundError(p)
        name = p.relative_to(root).as_posix()
        z.write(p,name)
        raw = p.read_bytes()
        binary = p.suffix == '.png'
        entries.append(dict(path=name,mode='100644',type='blob',
            content=base64.b64encode(raw).decode() if binary else raw.decode('utf-8'),
            binary=binary))
(release/'manifest.json').write_text(json.dumps(entries),encoding='utf-8')
print(f'Prepared {len(entries)} reviewed files; excluded legacy paper and outputs')
