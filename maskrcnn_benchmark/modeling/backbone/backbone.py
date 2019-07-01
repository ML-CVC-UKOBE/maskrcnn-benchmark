# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet
from . import efficientnet


@registry.BACKBONES.register("E-B0")
@registry.BACKBONES.register("E-B1")
@registry.BACKBONES.register("E-B2")
@registry.BACKBONES.register("E-B3")
@registry.BACKBONES.register("E-B4")
@registry.BACKBONES.register("E-B5")
@registry.BACKBONES.register("E-B6")
@registry.BACKBONES.register("E-B7")
def build_efficientnet_backbone(cfg):
    if cfg["MODEL"]["BACKBONE"]["CONV_BODY"] == "E-B0":
        name = 'efficientnet-b0'
        block_to_remove_stride = 11
    elif cfg["MODEL"]["BACKBONE"]["CONV_BODY"] == "E-B1":
        name = 'efficientnet-b1'
        block_to_remove_stride = 16
    elif cfg["MODEL"]["BACKBONE"]["CONV_BODY"] == "E-B2":
        name = 'efficientnet-b2'
        block_to_remove_stride = 16
    elif cfg["MODEL"]["BACKBONE"]["CONV_BODY"] == "E-B3":
        name = 'efficientnet-b3'
        block_to_remove_stride = 18
    elif cfg["MODEL"]["BACKBONE"]["CONV_BODY"] == "E-B4":
        name = 'efficientnet-b4'
        block_to_remove_stride = 22
    elif cfg["MODEL"]["BACKBONE"]["CONV_BODY"] == "E-B5":
        name = 'efficientnet-b5'
        block_to_remove_stride = 27
    else:
        raise NotImplementedError
    # print(name)
    body = efficientnet.EfficientNet.from_name(cfg, model_name=name)
    del body._conv_head
    del body._bn1
    del body._fc
    body._blocks[block_to_remove_stride]._depthwise_conv.stride = [1, 1]  # B0
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    print(model)
    return model


@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
