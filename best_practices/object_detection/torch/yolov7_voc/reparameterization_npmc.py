import argparse
from typing import *
import copy

import torch
import torch.nn as nn
import torch.fx as fx

def find_node(model, name):
    for idx, node in enumerate(list(model.graph.nodes)):
        if node.name == name:
            return node
    return None

# Fusing BatchNorm with Convolutional layer
def fuse_conv_bn_eval(conv, bn):
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = \
        fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_conv

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)

def _parent_name(target : str) -> Tuple[str, str]:
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name

def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, new_module)


def fuse(model: torch.nn.Module) -> torch.nn.Module:
    model = copy.deepcopy(model)
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    for node in fx_model.graph.nodes:
        if node.op != 'call_module': 
            continue
        if type(modules[node.target]) is nn.BatchNorm2d and type(modules[node.args[0].target]) is nn.Conv2d:
            if len(node.args[0].users) > 1:  
                continue
            conv = modules[node.args[0].target]
            bn = modules[node.target]
            fused_conv = fuse_conv_bn_eval(conv, bn)
            replace_node_module(node.args[0], modules, fused_conv)
            node.replace_all_uses_with(node.args[0])
            fx_model.graph.erase_node(node)
    fx_model.graph.lint()
    fx_model.recompile()
    return fx_model

# Fusing BatchNorm with Convolutional layers
def rep_conv1x1(model, head,conv1x1,conv3x3,tail):
    head.users = {conv3x3:None}
    conv3x3.users = {tail:None}
    tail._input_nodes = {conv3x3:None}
    tail._args = (conv3x3,)

    modules = dict(model.named_modules())
    weight_1x1_expanded = torch.nn.functional.pad(modules[conv1x1.target].weight, [1, 1, 1, 1])
    rbr_1x1_bias = modules[conv1x1.target].bias
    modules[conv3x3.target].weight = torch.nn.Parameter(modules[conv3x3.target].weight + weight_1x1_expanded)
    modules[conv3x3.target].bias = torch.nn.Parameter(modules[conv3x3.target].bias + rbr_1x1_bias)

# Fusing Implicit Knowledge layers
def rep_im(model, head, target, tail_1, tail_2, im_a_weight, im_m_weight):
    # detach
    head.users = {target:None}
    target._input_nodes = {head:None}
    target._args = (head,)
    target.users = {tail_1:None,tail_2:None}

    tail_1._input_nodes = target
    tail_1._args = (target,tail_1._args[1])

    tail_2._input_nodes.pop(tail_2._args[0])
    tail_2._input_nodes[target] = None
    temp_arg = list(tail_2._args)
    temp_arg[0] = target
    temp_arg = tuple(temp_arg)
    tail_2._args = temp_arg 

    # reparam
    modules = dict(model.named_modules())

    conv_weight = modules[target.target].weight
    c2,c1,_,_ = conv_weight.shape
    c2_,c1_,_,_ = im_a_weight.shape

    # ImplicitA
    with torch.no_grad():
        modules[target.target].bias += torch.matmul(conv_weight.reshape(c2,c1), im_a_weight.reshape(c1_,c2_)).squeeze(1)

    # ImplicitM
    with torch.no_grad():
        modules[target.target].bias *= im_m_weight.reshape(c2)
        modules[target.target].weight *= im_m_weight.transpose(0,1)

def fusing_yolov7(model):
    model = model.float().eval()
    # fusing conv-bn
    print("Fusing Conv-BN")
    fused_model = fuse(model)
    print("Fusing Conv-BN - Success")

    print("Reparameterizing")
    # head_0
    head = find_node(fused_model,'module_dict_module_dict_module_dict_model_75_act')
    conv1x1 = find_node(fused_model,'module_dict_module_dict_module_dict_model_102_rbr_1x1_0')
    conv3x3 = find_node(fused_model,'module_dict_module_dict_module_dict_model_102_rbr_dense_0')
    tail = find_node(fused_model,'module_dict_module_dict_module_dict_model_102_act')
    rep_conv1x1(fused_model,head,conv1x1,conv3x3,tail)

    head = find_node(fused_model,'module_dict_module_dict_module_dict_model_102_act')
    target = find_node(fused_model,'module_dict_module_dict_module_dict_model_105_m_0')
    tail_1 =  find_node(fused_model,'getattr_1')
    tail_2 = find_node(fused_model,'view')
    im_a_weight = getattr(getattr(fused_model.module_dict.module_dict.module_dict.model, "105").ia, "0")._parameters['NOTA_implicit']
    im_m_weight = getattr(getattr(fused_model.module_dict.module_dict.module_dict.model, "105").im, "0")._parameters['NOTA_implicit']
    rep_im(fused_model,head,target,tail_1,tail_2,im_a_weight,im_m_weight)

    # head_1
    head = find_node(fused_model,'module_dict_module_dict_module_dict_model_88_act')
    conv1x1 = find_node(fused_model,'module_dict_module_dict_module_dict_model_103_rbr_1x1_0')
    conv3x3 = find_node(fused_model,'module_dict_module_dict_module_dict_model_103_rbr_dense_0')
    tail = find_node(fused_model,'module_dict_module_dict_module_dict_model_103_act')
    rep_conv1x1(fused_model,head,conv1x1,conv3x3,tail)
    
    head = find_node(fused_model,'module_dict_module_dict_module_dict_model_103_act')
    target = find_node(fused_model,'module_dict_module_dict_module_dict_model_105_m_1')
    tail_1 =  find_node(fused_model,'getattr_2')
    tail_2 = find_node(fused_model,'view_1')
    im_a_weight = getattr(getattr(fused_model.module_dict.module_dict.module_dict.model, "105").ia, "1")._parameters['NOTA_implicit']
    im_m_weight = getattr(getattr(fused_model.module_dict.module_dict.module_dict.model, "105").im, "1")._parameters['NOTA_implicit']
    rep_im(fused_model,head,target,tail_1,tail_2,im_a_weight,im_m_weight)


    # head_2
    head = find_node(fused_model,'module_dict_module_dict_module_dict_model_101_act')
    conv1x1 = find_node(fused_model,'module_dict_module_dict_module_dict_model_104_rbr_1x1_0')
    conv3x3 = find_node(fused_model,'module_dict_module_dict_module_dict_model_104_rbr_dense_0')
    tail = find_node(fused_model,'module_dict_module_dict_module_dict_model_104_act')
    rep_conv1x1(fused_model,head,conv1x1,conv3x3,tail)

    head = find_node(fused_model,'module_dict_module_dict_module_dict_model_104_act')
    target = find_node(fused_model,'module_dict_module_dict_module_dict_model_105_m_2')
    tail_1 =  find_node(fused_model,'getattr_3')
    tail_2 = find_node(fused_model,'view_2')
    im_a_weight = getattr(getattr(fused_model.module_dict.module_dict.module_dict.model, "105").ia, "2")._parameters['NOTA_implicit']
    im_m_weight = getattr(getattr(fused_model.module_dict.module_dict.module_dict.model, "105").im, "2")._parameters['NOTA_implicit']
    rep_im(fused_model,head,target,tail_1,tail_2,im_a_weight,im_m_weight)

    print("Reparameterizing - Success")

    print("Recompile...")
    fused_model.graph.lint()
    fused_model.recompile()
    print("Success")
    return fused_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/root/workspace/projects/YOLOv7BestPractice/bestPractice3/yolov7/runs/train/finetuned_yolov7_voc_compressed_model_L2Norm_02/weights/best.pt', help='finetuned-yolov7 graphmodule path')
    parser.add_argument('--save_model_path', type=str, default='yolo7-voc-finetuned.pt', help='finetuned-yolov7 graphmodule path')
    args = parser.parse_args()
    
    model = torch.load(args.model,map_location='cpu')['model'].float().train()
    fused_model = fusing_yolov7(model)
    
    torch.save(fused_model,args.save_model_path)