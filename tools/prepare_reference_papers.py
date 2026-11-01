"""Fetch or copy three verified paper PDFs required by historical baseline constructors.

These PDFs are runtime provenance inputs, not training or evaluation data. They
stay untracked and outside release archives. The scientific Python is unchanged.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen

from common import ROOT

PAPERS = {
    'FLTrust_Cao_2021.pdf': (
        'https://www.ndss-symposium.org/wp-content/uploads/ndss2021_6C-2_24434_paper.pdf',
        'e569fcd8047337c74f7d00802419c840cf926bd13978a471ee0df21c72f4d8a5'),
    'RFA_Pillutla_2022.pdf': (
        'https://arxiv.org/pdf/1912.13445v2',
        '1875d0d6914a0f95394043743e000d057def68ee9a673e4ba5b1be9152a61cbc'),
    'Viceroy_Lewis_2023.pdf': (
        'https://jmlr.org/papers/volume24/22-0014/22-0014.pdf',
        '6b8140bc3f2ed3890485cda9e3d6a8b05b18667d5f4006b085f133b53dab746d'),
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--download', action='store_true', help='Download the three pinned public PDFs.')
    source.add_argument('--from-directory', type=Path, help='Copy verified PDFs from a local directory.')
    args = parser.parse_args()
    destination = ROOT / 'Experiment/resources'
    destination.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in PAPERS.items():
        target = destination / name
        if target.exists():
            if digest(target) != expected:
                raise ValueError('Existing PDF has unexpected bytes: ' + str(target))
            print('Verified existing:', name)
            continue
        with tempfile.NamedTemporaryFile(dir=destination, prefix='.reference-', suffix='.tmp', delete=False) as temp:
            temporary = Path(temp.name)
            try:
                if args.download:
                    request = Request(url, headers={'User-Agent': 'FedReTrust-reproduction/1.0'})
                    with urlopen(request, timeout=60) as response:
                        shutil.copyfileobj(response, temp)
                else:
                    with (args.from_directory / name).open('rb') as stream:
                        shutil.copyfileobj(stream, temp)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            if digest(temporary) != expected:
                raise ValueError('Downloaded/copied PDF differs from frozen fingerprint: ' + name)
            temporary.replace(target)
            print('Verified:', name)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
