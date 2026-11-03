"""Execute notebook using this Python, keeping Jupyter state in the workspace."""
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
for name, folder in [('IPYTHONDIR','ipython'), ('JUPYTER_CONFIG_DIR','config'),
                     ('JUPYTER_DATA_DIR','data'), ('JUPYTER_RUNTIME_DIR','runtime')]:
    path = root/'.jupyter-local'/folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(path)
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
nb = nbformat.read(root/'LearnLens_Colab.ipynb', as_version=4)
km = KernelManager(kernel_name='python3')
km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client = NotebookClient(nb, timeout=180, km=km, resources={'metadata': {'path': str(root)}})
client.execute()
nbformat.write(nb, root/'outputs/LearnLens_executed.ipynb')
print('Executed all',len(nb.cells),'notebook cells')
