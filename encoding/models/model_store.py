"""Model store which provides pretrained models."""
from __future__ import print_function
__all__ = ['get_model_file', 'purge']
import os
import zipfile

from ..utils import download, check_sha1

_model_sha1 = {name: checksum for checksum, name in [
    ('853f2fb07aeb2927f7696e166b215609a987fd44', 'resnet50'),
    ('5be5422ad7cb6a2e5f5a54070d0aa9affe69a9a4', 'resnet101'),
    ('6cb047cda851de6aa31963e779fae5f4c299056a', 'deepten_minc'),
    ('fc8c0b795abf0133700c2d4265d2f9edab7eb6cc', 'fcn_resnet50_ade'),
    ('eeed8e582f0fdccdba8579e7490570adc6d85c7c', 'fcn_resnet50_pcontext'),
    ('54f70c772505064e30efd1ddd3a14e1759faa363', 'psp_resnet50_ade'),
    ('558e8904e123813f23dc0347acba85224650fe5f', 'encnet_resnet50_ade'),
    ('7846a2f065e90ce70d268ba8ada1a92251587734', 'encnet_resnet50_pcontext'),
    ('6f7c372259988bc2b6d7fc0007182e7835c31a11', 'encnet_resnet101_pcontext'),
    ]}

encoding_repo_url = 'https://hangzh.s3.amazonaws.com/'
_url_format = '{repo_url}encoding/models/{file_name}.zip'

def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]

def get_model_file(name, root='./pretrain_models'):
    r"""Return location for the pretrained on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default './pretrain_models'
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    file_name = '{name}-{short_hash}'.format(name=name, short_hash=short_hash(name))
    root = os.path.expanduser(root)
    file_path = os.path.join(root, file_name+'.pth')
    sha1_hash = _model_sha1[name]
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            return file_path
        else:
            print('Mismatch in the content of model file detected. Downloading again.')
    else:
        print('Model file is not found. Downloading.')

    if not os.path.exists(root):
        os.makedirs(root)

    zip_file_path = os.path.join(root, file_name+'.zip')
    repo_url = os.environ.get('ENCODING_REPO', encoding_repo_url)
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'
    download(_url_format.format(repo_url=repo_url, file_name=file_name),
             path=zip_file_path,
             overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(root)
    os.remove(zip_file_path)

    if check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')

def purge(root='./pretrain_models'):
    r"""Purge all pretrained model files in local file store.

    Parameters
    ----------
    root : str, default './pretrain_models'
        Location for keeping the model parameters.
    """
    root = os.path.expanduser(root)
    files = os.listdir(root)
    for f in files:
        if f.endswith(".pth"):
            os.remove(os.path.join(root, f))

def pretrained_model_list():
    return list(_model_sha1.keys())
