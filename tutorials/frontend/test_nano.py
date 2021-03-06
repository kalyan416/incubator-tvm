import tvm
from tvm import relay
from tvm import autotvm

import numpy as np
from tvm.contrib import util, graph_runtime as runtime
from tvm.contrib.debugger import debug_runtime as rtime
from tvm import rpc
from tvm.contrib.download import download_testdata
from resnet50_3d import *
#from resnet_depth_merging import *
#from resnet_2d import *
#from resnet50_optimized import *
#from torch_q_utils import tune_tasks,tuning_rpc_option,tuning_option
from torch_q_utils import *
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
from python_utils import *

# PyTorch imports
import torch
import onnx
import torchvision.datasets as datasets
import torchvision
import time
#from torchsummaryX import summary


import logging
#logging.getLogger('autotvm').setLevel(logging.DEBUG)

demo = 'rpc'
quant = True

if demo == 'local':
	log_file1 = 'resnet-50_1d_f.log'
if demo == 'rpc':
	log_file1 = 'resnet-50_3d_f_nano_gpu.log'

#Loading pytorch pre-trained model in eval mode
model = resnet50()
#model = model.half()
model_dict = model.state_dict()
#print(summary(model,torch.zeros(1,3,224,224)))
m_state_dict = torch.load('/home/kalyan/models/pytorch/models/resnet50/resnet50_3d/fused_resnet50.pth')
#m_state_dict = torch.load('/home/kalyan/models/pytorch/models/resnet50/renet50_depth_merging/model_best_Sep_bn.pth.tar')
#m_state_dict = torch.load('/home/kalyan/models/pytorch/models/resnet50/resnet50_depth_merge_2D/model_best_Sep_withFC_bn.pth.tar')
#m_state_dict = torch.load('/home/kalyan/Downloads/resnet50_optimized_f.pth')
#m_state_dict = m_state_dict['state_dict']

#m_state_dict = {k:v for k, v in m_state_dict.items() if k[7:] in model_dict and model_dict[k[7:]].shape == v.shape}

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in m_state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model = model.eval()
model = model.cuda()
#model_name = 'resnet50'
#model = getattr(torchvision.models, model_name)(pretrained=True)
#model = torch.nn.Sequential(*(list(model.children())[:1]))
#model = model.eval()

# We grab the TorchScripted model via tracing
input_shape = [1, 3, 224, 224]
#input_data = torch.randn(input_shape)
input_data = torch.randn(input_shape).cuda()
#scripted_model = torch.jit.trace(model.half(), input_data.half()).eval() #Tvm Supports Trace models only in pytorch
scripted_model = torch.jit.trace(model, input_data).eval()
#torch.onnx.export(model.half(),input_data.half(),"test1.onnx")

from PIL import Image
img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_path = download_testdata(img_url, 'cat.png', module='data')
img = Image.open(img_path).resize((224, 224))

from torchvision import transforms
my_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
img = my_preprocess(img)
img = np.expand_dims(img, 0)
input_name = 'input.1'  # only one input, set it to this name
shape_list = [(input_name, img.shape)]
shape_dict = {input_name:img.shape}
#np.save('cat.npy',img)
'''f = open('cat.y','wb')
f.write(img.flatten())
f.close()'''
#onnx_model = onnx.load('test1.onnx')
#mod, params = relay.frontend.from_onnx(onnx_model, shape_dict,dtype='float16')
mod, params = relay.frontend.from_pytorch(scripted_model,
                                          shape_list)

if demo == 'rpc':
	print('RPC')
	set_cuda_target_arch('sm_53')
	tgt_cuda = tvm.target.cuda(model="nano")
	tgt_host="llvm -target=aarch64-linux-gnu"
	tgt = tgt_host
else :
	tgt = tvm.target.cuda()
	tgt_host="llvm"
	ctx = tvm.gpu(0)

if quant:
	print('quant')
	mod = quantize(mod, params, '/mnt/DATA/imageNet/val' ,data_aware=True)
	params = None


'''tasks = autotvm.task.extract_from_program(mod ,params , tgt,target_host=tgt_host,ops=(relay.op.get("nn.conv2d"),))
if demo == 'rpc':
	tune_tasks(tasks, **tuning_rpc_option)
else:
	tune_tasks(tasks, **tuning_option)'''

with autotvm.apply_history_best(log_file1):
	with relay.build_config(opt_level=3):
		graph, lib, params = relay.build(mod,target = tgt , target_host=tgt_host, params=params)

#print(lib.imported_modules[0].get_source())

#np.save('kernel0.npy',params['p0'].asnumpy())
#np.save('bias0.npy',params['p1'].asnumpy())

'''lib.export_library("deploy_lib.tar")
with open("deploy_graph.json", "w") as fo:
	fo.write(graph)
with open("deploy_param.params", "wb") as fo:
	fo.write(relay.save_param_dict(params))'''

if demo == 'rpc':
	tmp = util.tempdir()
	lib_fname = tmp.relpath('resnet50_3df.tar')
	lib.export_library(lib_fname)

	host = '192.168.1.92'
	port = 9090
	remote = rpc.connect(host, port,)

	# upload the library to remote device and load it
	remote.upload(lib_fname)
	rlib = remote.load_module('resnet50_3df.tar')

	ctx = remote.cpu(0)
	flib = rlib
else:
	flib = lib
#rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}

print("Compile...")
module = runtime.create(graph, flib, ctx)
#module = rtime.create(graph, flib, ctx)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
val_loader = torch.utils.data.DataLoader(datasets.ImageFolder('/mnt/DATA/imageNet/val', transforms.Compose([
					transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize,])),
					batch_size=1, shuffle=False,num_workers=1, pin_memory=True)
total = 0
top1 = 0
total_time = 0
module.set_input(**params)

start = time.time()
#top1 = AverageMeter('Acc@1', ':6.2f')
#top5 = AverageMeter('Acc@5', ':6.2f')
top1 = 0
top5 = 0
for i, (batch,target) in enumerate(val_loader):
	data = batch.cpu().numpy()
	#target = target.half()
	#np.save('input0.npy',data)
	#f = open('input0.y','wb')
	#f.write(data.flatten())
	#f.close()
	total = i+1
	module.set_input('input.1', tvm.nd.array(data.astype('float32')))
	module.run()
	prediction = module.get_output(0)
	#with torch.no_grad():
	#	output      = model(batch)
	#np.testing.assert_equal(prediction.asnumpy(), output.cpu().numpy())
	if np.argmax(prediction.asnumpy()[0]) == target.cpu().numpy()[0] :
		top1 = top1+1
		print(top1)
	if i > 0:  # only run inference on a few samples in this tutorial
		break
	#prediction = prediction.asnumpy()
	#prediction = torch.from_numpy(prediction)
	#acc1, acc5 = accuracy(prediction, target, topk=(1, 5))
	#top1.update(acc1[0], batch.size(0))
	#top5.update(acc5[0], batch.size(0))
	#print(' * total: {i}  Acc@1 {top1.avg:.6f} Acc@5 {top5.avg:.6f}'
        #      .format(i=i, top1=top1, top5=top5))
end = time.time()
ftimer = module.module.time_evaluator('run',ctx,1,100)
#module._run_debug()
prof_res = np.array(ftimer().results) * 1000 

print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
      (np.mean(prof_res), np.std(prof_res)))
print('total time :{}(sec)'.format(end-start))
print('total :{} top1 : {} accu: {}'.format(total,top1,top1/float(total)))
