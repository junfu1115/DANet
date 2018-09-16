import os
import os.path as osp
import logging
import time
import io


def create_logger(log_root_path,log_name):
    if not osp.exists(log_root_path):
        os.makedirs(log_root_path)
        assert osp.exists(log_root_path), '{} does not exist!!'.format(log_root_path)
    
    final_log_path = osp.join(log_root_path, log_name)
    if not osp.exists(final_log_path):
        os.makedirs(final_log_path)
        assert osp.exists(final_log_path), '{} does not exist!!'.format(final_log_path)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file = '{}_{}.log'.format(log_name,time.strftime("%Y-%m-%d-%H-%M",time.localtime()))
    BASIC_FORMAT = "%(asctime)s: %(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')
    fhlr = logging.FileHandler(osp.join(final_log_path, log_file))
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger
