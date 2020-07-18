import os
import cv2

import tvm
from tvm import relay
from tvm import autotvm

import numpy as np
from tvm.contrib import util, graph_runtime as runtime
from tvm.contrib.debugger import debug_runtime as rtime
from tvm import rpc
from tvm.contrib.download import download_testdata
from tvm.contrib import cc

#from peleenet import *
#from peleenet_1D import *
#from peleenet_depthmerged import *
#from peleenet_1_conv import *
#from peleenet_ahm import*
from peleenet_2M import*

from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
from python_utils import *
from torch_q_utils import *

import torch
import onnx
import torchvision.datasets as datasets
from torchvision import transforms
import torchvision
import time

import logging

from pelee_CC import Config
from pelee_data_augment import BaseTransform
from pelee_detection import Detect
#from pelee_prior_box import PriorBox
from prior_box_2M import PriorBox
from pelee_core import *
from data import  VOC_CLASSES
'''from configs.CC import Config
from layers.functions import Detect, PriorBox
from peleenet import build_net
from data import BaseTransform, VOC_CLASSES
from utils.core import *'''

#logging.getLogger('autotvm').setLevel(logging.DEBUG)
#cfg = Config.fromfile('/home/kalyan/libraries/Pelee.Pytorch/configs/Pelee_VOC.py')
cfg = Config.fromfile('Pelee_VOC_2M.py')

model = PeleeNet('test',304,cfg.model)
#model = SepBlock_nxn(3, 32)

#m_state_dict = torch.load('/home/kalyan/libraries/Pelee.Pytorch/weigths/Pelee_VOC.pth')
#m_state_dict = torch.load('/home/kalyan/libraries/Pelee.Pytorch/weigths/peleenet_1d/1d_Pelee_304_VOC.pth')
#m_state_dict = m_state_dict['state_dict']

demo = 'rpc'
quant = False
num_classes = cfg.model.num_classes
log_file1 = 'peleenet32_1D.log'

def cpu_soft_nms(boxes,sigma=0.5,Nt=0.3,threshold=0.001,method=1):
    N = boxes.shape[0]
    
    pos = 0
    maxscore = 0
    maxpos = 0
    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
	# get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1
	# add max box as a detection 
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]

	# swap ith box with position of max box
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
	# NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua #iou between max box and detection box

                    if method == 1: # linear
                        if ov > Nt: 
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > Nt: 
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight*boxes[pos, 4]
		    
		    # if box score falls below threshold, discard the box by swapping with last box
		    # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        boxes[pos,4] = boxes[N-1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    return keep

def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127
base = int(np.ceil(pow(num_classes, 1. / 3)))
colors = [_to_color(x, base)
          for x in range(num_classes)]
cats = [_.strip().split(',')[-1]
        for _ in open('data/coco_labels.txt','r').readlines()]
label_config = {'VOC': VOC_CLASSES, 'COCO': tuple(['__background__'] + cats)}
labels = label_config['VOC']

def draw_detection(im, bboxes, scores, cls_inds, fps, thr=0.2):
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = int(cls_inds[i])
        box = [int(_) for _ in box]
        thick = int((h + w) / 300)
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        cv2.putText(imgcv, mess, (box[0], box[1] - 7),
                    0, 1e-3 * h, colors[cls_indx], thick // 3)
        if fps >= 0:
            cv2.putText(imgcv, '%.2f' % fps + ' fps', (w - 160, h - 15),
                        0, 2e-3 * h, (255, 255, 255), thick // 2)

    return imgcv

'''from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in m_state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(m_state_dict)'''

#model = torch.nn.Sequential(*(list(model.children())[:1]))
model = model.eval()
#model = model.cuda()

input_shape = [1, 3, 1216, 608]
input_data = torch.randn(input_shape)
#input_data = input_data.half()
#torch.onnx.export(model.half(),input_data.half(),"peleenet_1D_depth.onnx")
scripted_model = torch.jit.trace(model, input_data).eval()
shape_dict = {'input.1':input_shape}
shape_list = [('input.1', input_shape)]
#onnx_model = onnx.load('peleenet_1D_depth.onnx')
#mod, params = relay.frontend.from_onnx(onnx_model, shape_dict,dtype='float16')
mod, params = relay.frontend.from_pytorch(scripted_model,
                                          shape_list)
                                          
if quant:
	print('quant')
	mod = quantize(mod, params, '/home/kalyan/libraries/Pelee.Pytorch/imgs/VOC' ,data_aware=True)
	params = None

if demo == 'rpc':
	print('RPC')
	set_cuda_target_arch('sm_53')
	tgt_cuda = tvm.target.cuda(model="nano")
	tgt_host="llvm -target=aarch64-linux-gnu"
	tgt = tgt_cuda
else :
	tgt = tvm.target.cuda()
	tgt_host="llvm"
	#tgt = tgt_host
	ctx = tvm.gpu(0)

'''tasks = autotvm.task.extract_from_program(mod ,params , tgt,target_host=tgt_host,ops=(relay.op.get("nn.conv2d"),))
if demo == 'rpc':
	tune_tasks(tasks, **tuning_rpc_option)
else:
	tune_tasks(tasks, **tuning_option)'''
#with autotvm.apply_history_best(log_file):
with relay.build_config(opt_level=3):
		graph, lib, params = relay.build(mod,target = tgt , target_host=tgt_host, params=params)
		
#print(graph)
#print(lib.imported_modules[0].get_source())

#lib.export_library("deploy_detect.so", cc.cross_compiler("aarch64-linux-gnu-gcc"))
'''lib.export_library("deploy_detect.so")
with open("deploy_detect.json", "w") as fo:
    fo.write(graph)
with open("deploy_detect.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))'''

if demo == 'rpc':
	tmp = util.tempdir()
	lib_fname = tmp.relpath('peleenet_3df.tar')
	lib.export_library(lib_fname)

	host = '192.168.1.98'
	port = 9090
	remote = rpc.connect(host, port,)

	# upload the library to remote device and load it
	remote.upload(lib_fname)
	rlib = remote.load_module('peleenet_3df.tar')

	ctx = remote.gpu(0)
	flib = rlib
else:
	flib = lib
	
module = runtime.create(graph, flib, ctx)
#module = rtime.create(graph, flib, ctx)
total = 0
total_time = 0
module.set_input(**params)

start = time.time()
top1 = AverageMeter('Acc@1', ':6.2f')
top5 = AverageMeter('Acc@5', ':6.2f')

_preprocess = BaseTransform(
    cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
im_path = '/home/kalyan/libraries/Pelee.Pytorch/imgs/VOC'
imgs_result_path = os.path.join(im_path, 'im_res')
#if not os.path.exists(imgs_result_path):
#    os.makedirs(imgs_result_path)

im_fnames = sorted((fname for fname in os.listdir(im_path)
                    if os.path.splitext(fname)[-1] == '.jpg'))
im_fnames = (os.path.join(im_path, fname) for fname in im_fnames)
anchor_config = anchors(cfg.model)
priorbox = PriorBox(anchor_config)
with torch.no_grad():
    priors = priorbox.forward()
    
'''f = open('prior_boxes.y','wb')
f.write(priors.cpu().numpy().flatten())
f.close()'''
detector = Detect(num_classes,
                  cfg.loss.bkg_label, anchor_config)

#top1 = 0
#top5 = 0
i = 0
for fname in im_fnames:
	print(fname)
	image = cv2.imread(fname, cv2.IMREAD_COLOR)
	w, h = image.shape[1], image.shape[0]
	scale = torch.Tensor([w, h, w, h])
	img = _preprocess(image).unsqueeze(0)
	'''f = open('image'+str(i)+'.y','wb')
	f.write(img.numpy().flatten())
	f.close()'''
	
	i = i+1
	break
	'''if i == 2:
		break
	else:
		continue	'''
	module.set_input('input.1', tvm.nd.array(input_data.cpu().numpy()))
	module.run()
	loc = module.get_output(0).asnumpy()
	conf = module.get_output(1).asnumpy()
	print(i)
	#if i==3:
	#	break
	loc = torch.from_numpy(loc).cpu().float()
	conf = torch.from_numpy(conf).cpu().float()
	boxes, scores = detector.forward(loc, conf, priors.cpu())
	boxes = (boxes[0] * scale).cpu().numpy()
	scores = scores[0].cpu().numpy()
	'''f = open('loc'+str(i)+'.y','wb')
	f.write(loc.numpy().flatten())
	f.close()
	f = open('conf'+str(i)+'.y','wb')
	f.write(conf.numpy().flatten())
	f.close()
	f = open('boxes'+str(i)+'.y','wb')
	f.write(boxes.flatten())
	f.close()
	f = open('scores'+str(i)+'.y','wb')
	f.write(scores.flatten())
	f.close()'''
	allboxes = []
	for j in range(1, num_classes):
		inds = np.where(scores[:, j] > cfg.test_cfg.score_threshold)[0]
		#print(inds.shape)
		#print(inds)
		if len(inds) == 0:
		    continue
		c_bboxes = boxes[inds]
		c_scores = scores[inds, j]
		c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
		    np.float32, copy=False)
		soft_nms = cfg.test_cfg.soft_nms
		# min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
		keep = cpu_soft_nms(c_dets, Nt=0.3)
		#keep = nms(c_dets, 0.3, force_cpu=soft_nms)
		keep = keep[:cfg.test_cfg.keep_per_class]
		c_dets = c_dets[keep, :]
		'''if i==2:
			f = open(str(j)+'.y','wb')
			f.write(c_dets.flatten())
			f.close()'''
		allboxes.extend([_.tolist() + [j] for _ in c_dets])
	allboxes = np.array(allboxes)
	boxes = allboxes[:, :4]
	scores = allboxes[:, 4]
	cls_inds = allboxes[:, 5]
	im2show = draw_detection(image, boxes, scores, cls_inds, -1, 0.5)
	if im2show.shape[0] > 1100:
		im2show = cv2.resize(im2show,
			                 (int(1000. * float(im2show.shape[1]) / im2show.shape[0]), 1000))
	if True:
		cv2.imwrite('test.jpg', im2show)
		cv2.imshow('test', im2show)
		cv2.waitKey(2000)

	#filename = os.path.join(imgs_result_path, '{}_stdn.jpg'.format(
	#	os.path.basename(fname).split('.')[0]))
	#cv2.imwrite(filename, im2show)
	
end = time.time()
print('time_evaluator')
ftimer = module.module.time_evaluator('run',ctx,1, 100)
#module._run_debug()
prof_res = np.array(ftimer().results) * 1000 

print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
      (np.mean(prof_res), np.std(prof_res)))
print('total time :{}(sec)'.format(end-start))
#print('total :{} top1 : {} accu: {}'.format(total,top1,top1/float(total))'''
