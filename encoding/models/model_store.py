"""Model store which provides pretrained models."""
from __future__ import print_function
__all__ = ['get_model_file', 'purge']
import os
import zipfile
import portalocker

from ..utils import download, check_sha1

_model_sha1 = {name: checksum for checksum, name in [
    # resnet
    ('25c4b50959ef024fcc050213a06b614899f94b3d', 'resnet50'),
    ('2a57e44de9c853fa015b172309a1ee7e2d0e4e2a', 'resnet101'),
    ('0d43d698c66aceaa2bc0309f55efdd7ff4b143af', 'resnet152'),
    # rectified
    ('9b5dc32b3b36ca1a6b41ecd4906830fc84dae8ed', 'resnet101_rt'),
    # resnest
    ('fb9de5b360976e3e8bd3679d3e93c5409a5eff3c', 'resnest50'),
    ('966fb78c22323b0c68097c5c1242bd16d3e07fd5', 'resnest101'),
    ('d7fd712f5a1fcee5b3ce176026fbb6d0d278454a', 'resnest200'),
    ('b743074c6fc40f88d7f53e8affb350de38f4f49d', 'resnest269'),
    # resnet other variants
    ('a75c83cfc89a56a4e8ba71b14f1ec67e923787b3', 'resnet50s'),
    ('03a0f310d6447880f1b22a83bd7d1aa7fc702c6e', 'resnet101s'),
    ('36670e8bc2428ecd5b7db1578538e2dd23872813', 'resnet152s'),
    # other segmentation backbones
    ('da4785cfc837bf00ef95b52fb218feefe703011f', 'wideresnet38'),
    ('b41562160173ee2e979b795c551d3c7143b1e5b5', 'wideresnet50'),
    # deepten paper
    ('1225f149519c7a0113c43a056153c1bb15468ac0', 'deepten_resnet50_minc'),
    # segmentation models
    ('662e979de25a389f11c65e9f1df7e06c2c356381', 'fcn_resnet50_ade'),
    ('eeed8e582f0fdccdba8579e7490570adc6d85c7c', 'fcn_resnet50_pcontext'),
    ('54f70c772505064e30efd1ddd3a14e1759faa363', 'psp_resnet50_ade'),
    ('075195c5237b778c718fd73ceddfa1376c18dfd0', 'deeplab_resnet50_ade'),
    ('5ee47ee28b480cc781a195d13b5806d5bbc616bf', 'encnet_resnet101_coco'),
    ('4de91d5922d4d3264f678b663f874da72e82db00', 'encnet_resnet50_pcontext'),
    ('9f27ea13d514d7010e59988341bcbd4140fcc33d', 'encnet_resnet101_pcontext'),
    ('07ac287cd77e53ea583f37454e17d30ce1509a4a', 'encnet_resnet50_ade'),
    ('3f54fa3b67bac7619cd9b3673f5c8227cf8f4718', 'encnet_resnet101_ade'),
    # resnest segmentation models
    ('2225f09d0f40b9a168d9091652194bc35ec2a5a9', 'deeplab_resnest50_ade'),
    ('06ca799c8cc148fe0fafb5b6d052052935aa3cc8', 'deeplab_resnest101_ade'),
    ]}

encoding_repo_url = 'https://hangzh.s3.amazonaws.com/'
_url_format = '{repo_url}encoding/models/{file_name}.zip'

def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]

def get_model_file(name, root=os.path.join('~', '.encoding', 'models')):
    r"""Return location for the pretrained on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    if name not in _model_sha1:
        from torchvision.models.resnet import model_urls
        if name not in model_urls:
            raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
        root = os.path.expanduser(root)
        return download(model_urls[name],
                        path=root,
                        overwrite=True)
    file_name = '{name}-{short_hash}'.format(name=name, short_hash=short_hash(name))
    root = os.path.expanduser(root)
    if not os.path.exists(root):
        os.makedirs(root)

    file_path = os.path.join(root, file_name+'.pth')
    sha1_hash = _model_sha1[name]

    lockfile = os.path.join(root, file_name + '.lock')
    with portalocker.Lock(lockfile, timeout=300):
        if os.path.exists(file_path):
            if check_sha1(file_path, sha1_hash):
                return file_path
            else:
                print('Mismatch in the content of model file {} detected.' +
                      ' Downloading again.'.format(file_path))
        else:
            print('Model file {} is not found. Downloading.'.format(file_path))

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

def purge(root=os.path.join('~', '.encoding', 'models')):
    r"""Purge all pretrained model files in local file store.

    Parameters
    ----------
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    """
    root = os.path.expanduser(root)
    files = os.listdir(root)
    for f in files:
        if f.endswith(".pth"):
            os.remove(os.path.join(root, f))

def pretrained_model_list():
    return list(_model_sha1.keys())
