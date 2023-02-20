import cv2
from glob import glob
import os
import numpy as np
from sklearn.metrics import jaccard_score
import collections

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
onnx_nms_imgs = sorted(glob(os.path.join('runs/onnx', '*.png')))
onnx_no_nms_imgs = sorted(glob(os.path.join('runs/onnx-no-nms', '*.png')))
onnx_torch_imgs = sorted(glob(os.path.join('runs/detect/exp6', '*.png')))

errors_nms = collections.deque([], 50)
errors_no_nms = collections.deque([], 50)

jacs_nms = collections.deque([], 50)
jacs_no_nms = collections.deque([], 50)

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (640*3, 360))
for nms, no_nms, torch in zip(onnx_nms_imgs, onnx_no_nms_imgs, onnx_torch_imgs):
    print(nms, no_nms)
    nms = cv2.imread(nms)
    no_nms = cv2.imread(no_nms)
    torch = cv2.imread(torch)
    
    err_nms = mse(torch, nms)
    err_no_nms = mse(torch, no_nms)
    errors_nms.append(err_nms)
    errors_no_nms.append(err_no_nms)
    
    jac_nms = jaccard_score(nms.flatten(),  torch.flatten(), average = 'macro')
    jac_no_nms = jaccard_score(no_nms.flatten(),  torch.flatten(), average = 'macro')
    jacs_nms.append(jac_nms)
    jacs_no_nms.append(jac_no_nms)
    
    print(f"MSE\nNMS:{np.mean(np.array(errors_nms))}\tNO_NMS:{np.mean(np.array(errors_no_nms))}")
    print(f"JACCARD\nNMS:{np.mean(np.array(jacs_nms))}\tNO_NMS:{np.mean(np.array(jacs_no_nms))}")
    cv2.imshow('images', np.hstack([nms, no_nms, torch]))
    out.write(np.hstack([nms, no_nms, torch]))
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
out.release()