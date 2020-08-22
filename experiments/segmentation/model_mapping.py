from collections import OrderedDict

import torch


# CLGD_key_map = {'head.gamma': 'head.clgd.gamma', 'head.skipconv.0.weight': 'head.clgd.skipconv.0.weight', 
#     'head.skipconv.1.weight': 'head.clgd.skipconv.1.weight', 'head.skipconv.1.bias': 'head.clgd.skipconv.1.bias', 
#     'head.skipconv.1.running_mean': 'head.clgd.skipconv.1.running_mean', 'head.skipconv.1.running_var': 'head.clgd.skipconv.1.running_var', 
#     'head.skipconv.1.num_batches_tracked': 'head.clgd.skipconv.1.num_batches_tracked', 'head.fusion.0.weight': 'head.clgd.fusion.0.weight', 
#     'head.fusion.1.weight': 'head.clgd.fusion.1.weight', 'head.fusion.1.bias': 'head.clgd.fusion.1.bias', 
#     'head.fusion.1.running_mean': 'head.clgd.fusion.1.running_mean', 'head.fusion.1.running_var': 'head.clgd.fusion.1.running_var', 
#     'head.fusion.1.num_batches_tracked': 'head.clgd.fusion.1.num_batches_tracked', 'head.fusion2.0.weight': 'head.clgd.fusion2.0.weight', 
#     'head.fusion2.1.weight': 'head.clgd.fusion2.1.weight', 'head.fusion2.1.bias': 'head.clgd.fusion2.1.bias', 
#     'head.fusion2.1.running_mean': 'head.clgd.fusion2.1.running_mean', 'head.fusion2.1.running_var': 'head.clgd.fusion2.1.running_var', 
#     'head.fusion2.1.num_batches_tracked': 'head.clgd.fusion2.1.num_batches_tracked', 'head.att.0.weight': 'head.clgd.att.0.weight', 
#     'head.att.0.bias': 'head.clgd.att.0.bias'}

del_keys = ["auxlayer.conv5.0.weight", "auxlayer.conv5.1.bias", "auxlayer.conv5.1.num_batches_tracked", \
    "auxlayer.conv5.1.running_mean", "auxlayer.conv5.1.running_var", "auxlayer.conv5.1.weight", \
    "auxlayer.conv5.4.bias", "auxlayer.conv5.4.weight"]

def _rename_glgd_weights(layer_keys):

    layer_keys = [k.replace("head.skipconv", "head.clgd.conv_low") for k in layer_keys]
    layer_keys = [k.replace("head.fusion2", "head.clgd.conv_out") for k in layer_keys]
    layer_keys = [k.replace("head.fusion", "head.clgd.conv_cat") for k in layer_keys]
    layer_keys = [k.replace("head.att", "head.clgd.conv_att") for k in layer_keys]
    layer_keys = [k.replace("head.gamma", "head.clgd.gamma") for k in layer_keys]

    return layer_keys

def _rename_dran_weights(layer_keys):

    layer_keys = [k.replace("head.conv5_s", "head.conv_cpam_b") for k in layer_keys]
    layer_keys = [k.replace("head.conv5_c", "head.conv_ccam_b") for k in layer_keys]
    layer_keys = [k.replace("head.conv51_c", "head.ccam_enc") for k in layer_keys]
    layer_keys = [k.replace("head.conv52", "head.conv_cpam_e") for k in layer_keys]
    layer_keys = [k.replace("head.conv51", "head.conv_ccam_e") for k in layer_keys]
    layer_keys = [k.replace("head.conv_f", "head.conv_cat") for k in layer_keys]
    layer_keys = [k.replace("head.conv6", "cls_seg") for k in layer_keys]
    layer_keys = [k.replace("head.conv7", "cls_aux") for k in layer_keys]
    
    layer_keys = [k.replace("head.en_s", "head.cpam_enc") for k in layer_keys]
    layer_keys = [k.replace("head.de_s", "head.cpam_dec") for k in layer_keys]
    layer_keys = [k.replace("head.de_c", "head.ccam_dec") for k in layer_keys]

    return layer_keys

def _rename_cpam_weights(layer_keys):

    layer_keys = [k.replace("head.cpam_dec.query_conv2", "head.cpam_dec.conv_query") for k in layer_keys]
    layer_keys = [k.replace("head.cpam_dec.key_conv2", "head.cpam_dec.conv_key") for k in layer_keys]
    layer_keys = [k.replace("head.cpam_dec.value2", "head.cpam_dec.conv_value") for k in layer_keys]

    return layer_keys

def rename_weight_for_head(weights):

    original_keys = sorted(weights.keys())
    layer_keys = sorted(weights.keys())

    new_weights = OrderedDict()
    for k in original_keys:
        v = weights[k]
        w=v

        layer_keys = _rename_glgd_weights(layer_keys)
        layer_keys = _rename_dran_weights(layer_keys)
        layer_keys = _rename_cpam_weights(layer_keys)
        key_map = {k: v for k, v in zip(original_keys, layer_keys)}
        new_weights[key_map[k] if key_map.get(k) else k] = w

    for keys in del_keys:
        del new_weights[keys]
    return new_weights