import os
import shutil
import argparse
import tarfile
from encoding.utils import download, mkdir

_TARGET_DIR = os.path.expanduser('~/.encoding/data')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize MINC dataset.',
        epilog='Example: python prepare_minc.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', type=str, default=None, help='dataset directory on disk')
    parser.add_argument('--no-download', action='store_true', help='disable automatic download if set')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite downloaded files if set, in case they are corrputed')
    args = parser.parse_args()
    return args

def download_minc(path, overwrite=False):
    _AUG_DOWNLOAD_URLS = [
        ('http://opensurfaces.cs.cornell.edu/static/minc/minc-2500.tar.gz', 'bcccbb3b1ab396ef540f024a5ba23eff54f7fe31')]
    download_dir = os.path.join(path, 'downloads')
    mkdir(download_dir)
    for url, checksum in _AUG_DOWNLOAD_URLS:
        filename = download(url, path=download_dir, overwrite=overwrite, sha1_hash=checksum)
        # extract
        with tarfile.open(filename) as tar:
            tar.extractall(path=path)

if __name__ == '__main__':
    args = parse_args()
    mkdir(os.path.expanduser('~/.encoding/datasets'))
    if args.download_dir is not None:
        if os.path.isdir(_TARGET_DIR):
            os.remove(_TARGET_DIR)
        os.symlink(args.download_dir, _TARGET_DIR)
    else:
        download_minc(_TARGET_DIR, overwrite=False)
