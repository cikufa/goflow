"""Audit pinned NVIDIA wheel sizes/metadata without downloading wheel payloads."""
import concurrent.futures
import email
import io
import json
import re
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

VERSION = '4.2.0.2'
LIMIT = 20_000_000_000

class RemoteWheel(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
    def seekable(self):
        return True
    def tell(self):
        return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset + (0 if whence == 0 else self.pos if whence == 1 else self.size)
        return self.pos
    def read(self, count=-1):
        end = self.size if count < 0 else min(self.size, self.pos + count)
        if end <= self.pos:
            return b''
        if end - self.pos > 64_000_000:
            raise RuntimeError('Metadata request exceeds 64 MB limit')
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end-1}'})
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status != 206:
                raise RuntimeError(f'Server ignored byte range: {self.url}')
            data = r.read(end-self.pos)
        self.pos += len(data)
        return data

def inspect(name):
    base = f'https://pypi.nvidia.com/{name}/'
    with urllib.request.urlopen(base, timeout=30) as r:
        page = r.read().decode()
    urls = re.findall(r'href="([^"]+)"', page)
    matches = [u for u in urls if f'{VERSION}-cp310-none-manylinux_2_34_x86_64.whl' in u]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one wheel for {name}: {matches}')
    url = urllib.parse.urljoin(base, matches[0].split('#')[0])
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
        size = int(r.headers['Content-Length'])
    with zipfile.ZipFile(RemoteWheel(url, size)) as z:
        metadata_path = f"{name.replace('-', '_')}-{VERSION}.dist-info/METADATA"
        meta = email.message_from_bytes(z.read(metadata_path))
    return {'name': name, 'url': url, 'bytes': size, 'requires': meta.get_all('Requires-Dist', [])}

def main():
    pending = {'isaacsim', 'isaacsim-extscache-physics', 'isaacsim-extscache-kit', 'isaacsim-extscache-kit-sdk'}
    records = {}
    while pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            batch = list(pool.map(inspect, sorted(pending)))
        for item in batch:
            records[item['name']] = item
            print(item['name'], item['bytes'], flush=True)
        pending = set()
        for item in batch:
            for dep in item['requires']:
                name = re.split(r'[\s(<>=;\[]', dep)[0]
                if name.startswith('isaacsim-') and name not in records:
                    pending.add(name)
    report = {'version': VERSION, 'wheels': list(records.values()), 'total_nvidia_wheel_bytes': sum(x['bytes'] for x in records.values())}
    Path('.cache/isaac-download-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print('TOTAL NVIDIA WHEELS', report['total_nvidia_wheel_bytes'])
    print('Other Python dependencies and runtime assets are additional.')
    if report['total_nvidia_wheel_bytes'] > LIMIT:
        raise SystemExit('Over 20 GB: approval required before payload download')

if __name__ == '__main__':
    main()
