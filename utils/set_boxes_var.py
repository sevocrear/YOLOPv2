import os
from argparse import ArgumentParser
parser = ArgumentParser()
    
parser.add_argument('--width', type=int, default=384, help='image width')
parser.add_argument('--height', type=int, default=640, help='image height')
parser.add_argument('--anchors-num', type=int, default=3, help='Number of anchors')
args = parser.parse_args()
boxes = str(int(args.anchors_num*(args.width*args.height/64 + args.width*args.height/(16*16) + args.width*args.height/(32*32))))
print(boxes)
