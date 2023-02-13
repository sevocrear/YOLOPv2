#! /usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnxsim import simplify
from argparse import ArgumentParser

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, cxcywh):
        x1 = (cxcywh[..., 0] - cxcywh[..., 2] / 2)[..., np.newaxis]  # top left x
        y1 = (cxcywh[..., 1] - cxcywh[..., 3] / 2)[..., np.newaxis]  # top left y
        x2 = (cxcywh[..., 0] + cxcywh[..., 2] / 2)[..., np.newaxis]  # bottom right x
        y2 = (cxcywh[..., 1] + cxcywh[..., 3] / 2)[..., np.newaxis]  # bottom right y
        y1x1y2x2 = torch.cat([y1,x1,y2,x2], dim=2)
        return y1x1y2x2


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-o',
        '--opset',
        type=int,
        default=11,
        help='onnx opset'
    )
    parser.add_argument(
        '-b',
        '--batches',
        type=int,
        default=1,
        help='batch size'
    )
    parser.add_argument('--width', type=int, default=384, help='image width')
    parser.add_argument('--height', type=int, default=640, help='image height')
    parser.add_argument('--anchors-num', type=int, default=3, help='Number of anchors')
    
    args = parser.parse_args()

    model = Model()

    MODEL = f'cxcywh_y1x1y2x2'
    OPSET=args.opset
    BATCHES = args.batches
    BOXES = int(args.anchors_num*(args.width*args.height/64 + args.width*args.height/(16*16) + args.width*args.height/(32*32)))
    onnx_file = f"data/graphs/{MODEL}.onnx"
    cxcywh = torch.randn(BATCHES, BOXES, 4)

    torch.onnx.export(
        model,
        args=(cxcywh),
        f=onnx_file,
        opset_version=OPSET,
        input_names = ['cxcywh'],
        output_names=['y1x1y2x2'],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
