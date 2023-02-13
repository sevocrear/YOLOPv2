import onnx
import torch
import torch.nn as nn
import numpy as np
from onnxsim import simplify
import argparse


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def _make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def forward(self, pred):
        anchor_grid = np.load('data/anchors/model_105_anchor_grid.npy')
        anchor_grid = torch.tensor(anchor_grid, dtype=torch.float32).to(DEVICE)
        z = []
        st = [8,16,32]
        for i in range(3):
            bs, _, ny, nx = pred[i].shape
            pred[i] = pred[i].view(bs, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            y = pred[i].sigmoid()
            gr = self._make_grid(nx, ny).to(pred[i].cpu())
            
            y0 = (y[..., 0:1] * 2. - 0.5 + gr[..., 0:1]) * st[i]  # xy
            y1 = (y[..., 1:2] * 2. - 0.5 + gr[..., 1:2]) * st[i]  # xy
            y2 = (y[..., 2:3] * 2) ** 2 * anchor_grid[i][..., 0:1]  # wh
            y3 = (y[..., 3:4] * 2) ** 2 * anchor_grid[i][..., 1:2]  # wh
            y4 = y[..., 4:]
            y5 = torch.cat([y0,y1,y2,y3,y4], dim=4)
            z.append(y5.view(bs, -1, 85))

        pred = torch.cat(z, 1)
        return pred

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=384, help='image width')
    parser.add_argument('--height', type=int, default=640, help='image height')
    return parser

if __name__ == "__main__":
    opt = make_parser().parse_args()
    DEVICE='cpu'
    RESOLUTION = [
        [192,320],
        [256,320],
        [256,416],
        [288,480],
        [384,640],
        [480,640],
        [480,800],
        [384,1280],
        [736,1280],
    ]
    MODEL = f'split_for_trace_model'
    model = Model()
    model.to(DEVICE)
    H = opt.height
    W = opt.width
    if [W, H] in RESOLUTION:
        pred = [
            torch.ones([1,255,H//8,W//8], dtype=torch.float32).to(DEVICE),
            torch.ones([1,255,H//16,W//16], dtype=torch.float32).to(DEVICE),
            torch.ones([1,255,H//32,W//32], dtype=torch.float32).to(DEVICE),
        ]
        onnx_file = f'data/graphs/{MODEL}_{H}x{W}.onnx'
        torch.onnx.export(
            model,
            args=(pred),
            f=onnx_file,
            opset_version=11,
            input_names=[
                'split_for_trace_model_pred0',
                'split_for_trace_model_pred1',
                'split_for_trace_model_pred2',
            ],
            output_names=[
                'split_for_trace_model_pred',
            ],
        )
        model_onnx1 = onnx.load(onnx_file)
        model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
        onnx.save(model_onnx1, onnx_file)

        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)
        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)
        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)
