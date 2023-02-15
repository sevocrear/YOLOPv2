import argparse
import onnx
import torch
import os

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='data/weights/yolopv2.pt', help='model.pt path')
    parser.add_argument('--onnx', type=str, default='data/graphs/', help='ONNX output path')
    parser.add_argument('--width', type=int, default=640, help='input width')
    parser.add_argument('--height', type=int, default=384, help='input height')
    return parser

def load_torch_model():
    '''
    Load weights
    '''
    # Load model
    model  = torch.jit.load(opt.weights)
    model = model.cpu()
    return model

def convert_2_onnx(model):
    '''
    Convert PyTorch Model to ONNX
    '''
    onnx_file = f"yolopv2_{opt.height}x{opt.width}.onnx"
    x = torch.randn(1, 3, opt.height, opt.width).cpu()
    torch.onnx.export(
        model,
        args=(x),
        f=os.path.join(opt.onnx,onnx_file),
        opset_version=11,
        input_names=['input'],
        output_names=['pred','anchor_grid0','anchor_grid1','anchor_grid2','seg','ll'],
        do_constant_folding=False,
    )
    model_onnx1 = onnx.load(os.path.join(opt.onnx,onnx_file))
    onnx.checker.check_model(model_onnx1)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, os.path.join(opt.onnx,onnx_file))
    print(f'ONNX is saved on path {os.path.join(opt.onnx,onnx_file)}')
    
if __name__ == '__main__':
    opt =  make_parser().parse_args()
    print(opt)
    
    model = load_torch_model()
    
    convert_2_onnx(model)
