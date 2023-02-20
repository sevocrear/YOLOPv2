#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime
from glob import glob
import os
def run_inference(onnx_session, input_size, image, score_th=0.5):
    image_width, image_height = image.shape[1], image.shape[0]
    # Pre process:Resize, BGR->RGB, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')
    input_image = input_image / 255.0

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    results = onnx_session.run(None, {input_name: input_image})

    # Post process
    drivable_area = np.squeeze(results[0])
    lane_line = np.squeeze(results[1])
    scores = results[2]
    batchno_classid_y1x1y2x2 = results[3]

    # Drivable Area Segmentation
    drivable_area = drivable_area.transpose(1, 2, 0)
    drivable_area = cv.resize(
        drivable_area,
        dsize=(image_width, image_height),
        interpolation=cv.INTER_LINEAR,
    )
    drivable_area = drivable_area.transpose(2, 0, 1)

    # Lane Line
    lane_line = cv.resize(
        lane_line,
        dsize=(image_width, image_height),
        interpolation=cv.INTER_LINEAR,
    )

    # Traffic Object Detection
    od_bboxes, od_scores, od_class_ids = [], [], []
    for score, batchno_classid_y1x1y2x2_ in zip(
            scores,
            batchno_classid_y1x1y2x2,
    ):
        class_id = int(batchno_classid_y1x1y2x2_[1])

        if score_th > score:
            continue
        y1 = batchno_classid_y1x1y2x2_[-4]
        x1 = batchno_classid_y1x1y2x2_[-3]
        y2 = batchno_classid_y1x1y2x2_[-2]
        x2 = batchno_classid_y1x1y2x2_[-1]
        y1 = int(y1 * (image_height / input_size[0]))
        x1 = int(x1 * (image_width / input_size[1]))
        y2 = int(y2 * (image_height / input_size[0]))
        x2 = int(x2 * (image_width / input_size[1]))

        od_bboxes.append([x1, y1, x2, y2])
        od_class_ids.append(class_id)
        od_scores.append(score)

    return drivable_area, lane_line, od_bboxes, od_scores, od_class_ids

def infer(frame, input_size, onnx_session, start_time):
    debug_image = copy.deepcopy(frame)

    # Inference execution
    drivable_area, lane_line, bboxes, scores, class_ids = run_inference(
        onnx_session,
        input_size,
        frame,
    )

    elapsed_time = time.time() - start_time

    # Draw
    debug_image = draw_debug(
        debug_image,
        elapsed_time,
        drivable_area,
        lane_line,
        bboxes,
        scores,
        class_ids,
    )
    return debug_image

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--images", type=str, default="data/demo/")
    parser.add_argument("--output", type=str, default="runs/onnx/")
    parser.add_argument(
        "--model",
        type=str,
        default='data/graphs/yolopv2_post_384x640.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='384,640',
    )

    args = parser.parse_args()
    model_path = args.model
    input_size = args.input_size

    input_size = [int(i) for i in input_size.split(',')]

    # Initialize video capture
    cap_device = args.device
    input = 'video'
    os.makedirs(args.output, exist_ok= True)
    if args.movie is not None:
        cap_device = args.movie
    if cap_device:
        cap = cv.VideoCapture(cap_device)
    elif args.images:
        input = 'images'
        images = glob(os.path.join(args.images, '*.png')) + glob(os.path.join(args.images, '*.jpg'))
        images.sort()
        images = iter(images)
    elif args.image:
        input = 'image'
    else:
        exit()
    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )
    print('device:', onnxruntime.get_device())
    while True:

        # Capture read
        if input == 'video':
            ret, frame = cap.read()
            if not ret:
                break
            debug_image = infer(frame, input_size, onnx_session, start_time)
        elif input == 'image':
            frame = cv.imread(args.image, cv.IMREAD_UNCHANGED)
            start_time = time.time()
            _ = infer(frame, input_size, onnx_session, start_time)
            start_time = time.time()
            debug_image = infer(frame, input_size, onnx_session, start_time)
            cv.imwrite(os.path.join(args.output, f"{len(glob(os.path.join(args.output, '*.png')))}.png"), debug_image)
            break
        elif input == 'images':
            frame = cv.imread(next(images), cv.IMREAD_UNCHANGED)
            start_time = time.time()
            debug_image = infer(frame, input_size, onnx_session, start_time)
            print(f"infered {len(glob(os.path.join(args.output, '*.png')))}")
            cv.imwrite(os.path.join(args.output, f"{len(glob(os.path.join(args.output, '*.png')))}.png"), debug_image)

        # key = cv.waitKey(1)
        # if key == 27:  # ESC
        #     break
        # cv.imshow('YOLOP v2', debug_image)
    if input == 'video':
        cap.release()
        cv.destroyAllWindows()


def draw_debug(
    debug_image,
    elapsed_time,
    drivable_area,
    lane_line,
    bboxes,
    scores,
    class_ids,
):

    # Drivable Area
    mask = np.where(drivable_area[1] > 0.5, 0, 1)
    debug_image[mask==0] = debug_image[mask==0]*0.5 + np.array([0, 255, 0])*0.5

    # Draw:Lane Line
    mask = np.where(lane_line > 0.5, 0, 1)
    debug_image[mask==0] = debug_image[mask==0]*0.5 + np.array([0, 0, 255])*0.5
    # Draw:Traffic Object Detection
    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        x1, y1 = int(bbox[0]), int(bbox[1])
        x2, y2 = int(bbox[2]), int(bbox[3])

        cv.rectangle(debug_image, (x1, y1), (x2, y2), [0,255,255], thickness=2, lineType=cv.LINE_AA)
        # cv.putText(debug_image, '%d:%.2f' % (class_id, score), (x1, y1 - 5), 0,
        #            0.7, (255, 255, 0), 2)

    # Inference elapsed time
    # cv.putText(debug_image,
    #            "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
    #            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
    #            cv.LINE_AA)

    return debug_image


if __name__ == '__main__':
    main()