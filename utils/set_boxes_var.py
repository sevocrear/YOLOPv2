import os
from argparse import ArgumentParser
parser = ArgumentParser()
    
parser.add_argument('--width', type=int, default=640, help='image width')
parser.add_argument('--height', type=int, default=384, help='image height')
parser.add_argument('--anchors-num', type=int, default=3, help='Number of anchors')
args = parser.parse_args()
boxes = str(int(args.anchors_num*(args.width*args.height/64 + args.width*args.height/(16*16) + args.width*args.height/(32*32))))

boxes1 = int(args.width*args.height/8**2*args.anchors_num)
boxes2 = int(args.width*args.height/8**3*args.anchors_num*2)
boxes3 = int(args.width*args.height/8**4*args.anchors_num*4)
print(boxes1, boxes2, boxes3, sep = ' ')