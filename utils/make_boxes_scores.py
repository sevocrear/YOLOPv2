#! /usr/bin/env python

import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
from argparse import ArgumentParser

"""
prediction [1, 5040, 85]

80 classes

85

[0] -> center_x
[1] -> center_y
[2] -> width
[3] -> height
[4] -> box_score
[5]-[84] -> class_score
"""


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        boxes = x[..., :4] # xywh [n, boxes, 4]
        box_scores = x[..., 4:5] # [n, boxes, 1]
        class_scores = x[..., 5:] # [n, boxes, 80]
        # scores = box_scores * class_scores
        # scores = scores.permute(0,2,1)
        nms_scores = box_scores * class_scores
        nms_scores = nms_scores.permute(0,2,1)
        class_scores = class_scores.permute(0,2,1)

        return boxes, nms_scores, class_scores


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
    
    parser.add_argument(
        '-c',
        '--classes',
        type=int,
        default=80,
        help='classes'
    )
    args = parser.parse_args()

    model = Model()

    MODEL = f'boxes_scores'
    OPSET=args.opset
    BATCHES = args.batches
    BOXES = int(args.anchors_num*(args.width*args.height/64 + args.width*args.height/(16*16) + args.width*args.height/(32*32)))
    CLASSES = args.classes

    onnx_file = f"data/graphs/{MODEL}.onnx"
    x = torch.randn(BATCHES, BOXES, CLASSES+5)

    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=OPSET,
        input_names = ['predictions'],
        output_names=['boxes_cxcywh','scores', 'class_scores'],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

