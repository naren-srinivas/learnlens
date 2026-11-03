"""Embed the actual benchmark in the offline demo; never hand-copy results."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
payload = (root/'outputs/benchmark.json').read_text(encoding='utf-8')
data = json.loads(payload)
path = root/'docs/index.html'
html = path.read_text(encoding='utf-8')
embedded = json.dumps(data, separators=(',',':')).replace('</', '<\\/')
html, count = re.subn(r'/\* BENCHMARK_START \*/.*?/\* BENCHMARK_END \*/',
    lambda _: '/* BENCHMARK_START */\nconst BENCHMARK = '+embedded+';\n/* BENCHMARK_END */',
    html, flags=re.S)
assert count == 1
path.write_text(html,encoding='utf-8')
(root/'docs/benchmark.json').write_text(payload,encoding='utf-8')
print(f"Exported {len(data['records'])} measured fits")
