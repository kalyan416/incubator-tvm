#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>

#include"thread_handler.hpp"

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>


#include <fstream>
#include <iterator>
#include <algorithm>

#include"peleenet_utills.hpp"
#include <chrono>
#include<string>
#include<dirent.h>

#define MAX_BUF_ID 5
#define MAX_THREADS 4

using namespace std;
using namespace cv;

void init(int argc, char** argv);

int main(int argc, char** argv)
{
	if(argc != 4)
	{
		return 0;
	}
	init(argc,argv); 
}


float * p_boxes;
float variance[2];
float Mean[3];
float Cmean[3];
float sd[3];
dispatch_queue *q;
std::queue<int> free_buff;
std::queue<cv::Mat> pro_buf;
int max_in_images;
std::string img_paths, lib_path;
bool flag_detect = false;
int batch_size;


bool preProcess_peleenet(dispatch_queue &op, int buf_id);
bool inference_peleenet(dispatch_queue &op, int buf_id);
bool postProcess_peleenet(dispatch_queue &op, int buf_id);
bool Display_peleenet(dispatch_queue &op, int buf_id);
bool producer_peleenet(dispatch_queue &op, int buf_id);

void init(int argc, char** argv)
{
	lib_path = argv[1];
     	img_paths = argv[2];
     	int dtype_code = kDLFloat;
	int dtype_bits = 32;
	int dtype_lanes = 1;
	int device_type = kDLGPU;
	int device_id = 0;
	batch_size = 1;
     	//tvm::runtime::PackedFunc set_input, run, get_output;
     	q = new dispatch_queue(" peleenet Demo", MAX_THREADS);
	q->max_threads = MAX_THREADS;
	q->unlock_main = false;
	q->dead_cnt = 0;
	max_in_images = batch_size * 10 + 1;
	for(int i=0; i<MAX_BUF_ID; i++)
	{
		free_buff.push(i);
	}
     	std::string file_path;
     	if(atoi(argv[3]))
     	{
     		flag_detect = true;
     	}
     	// parameters need to be TVMByteArray type to indicate the binary data
	TVMByteArray params_arr;
	std::string params_data_cpy;
	tvm::runtime::Module mod;
     	if(flag_detect)
     	{
     		file_path = lib_path + "deploy_detect.so";
 		// tvm module for compiled functions
		tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(file_path.c_str());
	
		file_path = lib_path + "deploy_detect.json";
		// json graph
		std::ifstream json_in(file_path.c_str(), std::ios::in);
		std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
		json_in.close();
	
		file_path = lib_path + "deploy_detect.params";
		// parameters in binary
		std::ifstream params_in(file_path.c_str(), std::ios::binary);
		std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
		params_data_cpy = params_data;
		params_in.close();
		params_arr.data = params_data_cpy.c_str();
		params_arr.size = params_data_cpy.length();
		// get global function module for graph runtime
		mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);
	}
	else
	{
     		file_path = lib_path + "deploy_classify.so";
 		// tvm module for compiled functions
		tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(file_path.c_str());
	
		file_path = lib_path + "deploy_classify.json";
		// json graph
		std::ifstream json_in(file_path.c_str(), std::ios::in);
		std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
		json_in.close();
	
		file_path = lib_path + "deploy_classify.params";
		// parameters in binary
		std::ifstream params_in(file_path.c_str(), std::ios::binary);
		std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
		params_in.close();
		params_data_cpy = params_data;
		params_arr.data = params_data_cpy.c_str();
		params_arr.size = params_data_cpy.length();
		// get global function module for graph runtime
		mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);
	}
	
	if(flag_detect)
	{
		std::vector<std::vector<float>> var ;
		get_min_max_sizes(10,90,304,6,var);
		std::vector<int> feature_maps{38, 19, 10, 5, 3, 1};
		std::vector<int> steps{8, 16, 30, 60, 101, 304};
		std::vector<float> aspect_ratios{2, 3} ;
	
		p_boxes = priorbox_generation(608, feature_maps, steps, var, aspect_ratios, 23280);
		variance[0] = 0.1;
		variance[1] = 0.2;
	
		Mean[0] = 103.94;
		Mean[1] = 116.78;
		Mean[2] = 123.68;
	}
	else
	{
		Cmean[0] = 0.485;
		Cmean[1] = 0.456;
		Cmean[2] = 0.406;
		
		sd[0] = 0.229;
		sd[1] = 0.224;
		sd[2] = 0.225;
	}
	q->set_input = mod.GetFunction("set_input");
	
	//get the function from the module(load patameters)
	tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
	load_params(params_arr);
	
	q->run = mod.GetFunction("run");
	q->get_output = mod.GetFunction("get_output");
	
	int in_ndim = 4;
	int64_t in_shape[4] = {1, 3, 1216, 608};
	
	int out_ndim = 3;
	int64_t out_shape[3] = {1,23280, 4};
	
	int out_ndim1 = 2;
	int64_t out_shape1[3] = {23280, 21};
	if(!flag_detect)
	{
		in_shape[2] = 224;
		in_shape[3] = 224;
		out_ndim = 2;
		out_shape[0] = 1;
		out_shape[1] = 1000;
	}
	q->out_put.resize(MAX_BUF_ID);
	q->disp_images.resize(MAX_BUF_ID);
	q->in_P.resize(MAX_BUF_ID);
	q->out_bd.resize(MAX_BUF_ID);
	q->out_con.resize(MAX_BUF_ID);
	q->sizes.resize(MAX_BUF_ID);
	for(int i=0; i<MAX_BUF_ID; i++)
	{
		TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, kDLCPU, device_id, &q->in_P[ i ]);
		//in_P[i] = tmp_t;
		//if(flag_detect)
		//	out_shape[2] = 4;
		TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, kDLCPU, device_id, &q->out_bd[ i ]);
		//out_bd[i] = tmp_t;
		if(flag_detect)
		{
			//out_shape[2] = 21;
			TVMArrayAlloc(out_shape1, out_ndim1, dtype_code, dtype_bits, dtype_lanes, kDLCPU, device_id, &q->out_con[ i ]);
		}
		//out_con[i] = tmp_t;
	}
	std::function<bool(dispatch_queue&,int)> fn1 = preProcess_peleenet;
	q->dispatch(fn1,0);
	std::function<bool(dispatch_queue&,int)> fn2 = producer_peleenet;
	q->dispatch(fn2,0);
	//preProcess_peleenet(*q, 0);
	//inference_peleenet(*q, 0);
	//postProcess_peleenet(*q, 0);
	//Display_peleenet(*q, 0);
	std::unique_lock<std::mutex> lock_id(q->lock_main_);
	q->wait_main.wait(lock_id,[]{return q->unlock_main;});
	cout<<"main unlocked"<<endl;
	lock_id.unlock();
}
//float total_time = 0.0f;
//unsigned int cnt = 0;
bool producer_peleenet(dispatch_queue &op, int buf_id)
{
	std::unique_lock<std::mutex> lock_id(op.lock_producer);
	DIR *dir;
	struct dirent *ent;
	cv::Mat img;
	string img_path;
	dir = opendir(img_paths.c_str());
	while(1)
	{
		if((ent = readdir(dir)) == NULL)
 		{
 			closedir(dir);
 			std::function<bool(dispatch_queue&,int)> fn1 = preProcess_peleenet;
 			op.wait_producer.wait(lock_id,[]{return pro_buf.size() == 0;});
 			op.wait_producer.wait(lock_id,[]{return free_buff.size() == MAX_BUF_ID;});
			op.dispatch(fn1, -1);
			op.dead_cnt = op.dead_cnt + 1;
			batch_size = 0;
			op.app_input.notify_all();
			break;
 		}
 		if(ent->d_name[0] == '.')
 		{
 			continue;
 		}
 		img_path = img_paths + ent->d_name;
 		img = cv::imread(img_path.c_str(), 1);
 		op.wait_producer.wait(lock_id,[]{return pro_buf.size() <max_in_images;});
 		pro_buf.push(img);
 		op.app_input.notify_all();
	}
	lock_id.unlock();
	return true;
}


float total_time = 0.0f;
float pre_total=0.0f, inf_total=0.0f, post_total=0.0f;
unsigned int cnt = 0;
std::chrono::system_clock::time_point prev_time;

bool preProcess_peleenet(dispatch_queue &op, int buf_id)
{
	std::unique_lock<std::mutex> lock_id(op.lock_pre);
	if(buf_id == -1)
	{
		std::function<bool(dispatch_queue&,int)> fn1 = inference_peleenet;
		op.dispatch(fn1, -1);
		op.dead_cnt = op.dead_cnt + 1;
		if(op.dead_cnt == MAX_THREADS)
		{
			cout<<"avg time:"<<(total_time/(cnt-1))<<"(ms)"<<endl;
			cout<<"pre avg time:"<<(pre_total/cnt)<<"(ms)"<<endl;
			cout<<"inf avg time:"<<(inf_total/cnt)<<"(ms)"<<endl;
			cout<<"post avg time:"<<(post_total/cnt)<<"(ms)"<<endl;
			op.unlock_main = true;
			op.wait_main.notify_all();
		}
		return true;
	}
	else
	{
		auto pe1 = std::chrono::high_resolution_clock::now();
		
		op.app_input.wait(lock_id,[]{return free_buff.size();});
		op.app_input.wait(lock_id,[]{return pro_buf.size() >= batch_size;});
		buf_id = std::move(free_buff.front());
		vector<cv::Mat> images;
		for(int i=0; i<batch_size; i++)
		{
			images.push_back(std::move(pro_buf.front()));
			pro_buf.pop();
		}
		op.wait_producer.notify_all();
		free_buff.pop();
    		//op.disp_images[buf_id].clear();
    		if(batch_size)
    		{
    			if(flag_detect)
    			{
				peleeNet_pre_pocess(static_cast<float*>(op.in_P[buf_id]->data), images, Mean, op.sizes[buf_id], true, 1, 608);
			}
			else
			{
				classifer_pre_process(static_cast<float*>(op.in_P[buf_id]->data), images, Cmean, sd);
			}
		}
		else
			buf_id = -1;
		
		/*auto t1 = std::chrono::high_resolution_clock::now();
		float duration = 0.0f;
		if(cnt)
			duration = std::chrono::duration_cast<std::chrono::microseconds>( t1 -  prev_time).count() / 1000;
		prev_time = t1;
		total_time += duration;
		cnt++;*/
		
		auto pe2 = std::chrono::high_resolution_clock::now();
		auto pe_duration = std::chrono::duration_cast<std::chrono::microseconds>( pe2 -  pe1).count() / 1000;
		pre_total += pe_duration;
		
		std::function<bool(dispatch_queue&,int)> fn1 = inference_peleenet;
		op.dispatch(fn1, buf_id);
		if(MAX_THREADS > 1)
		{
			std::function<bool(dispatch_queue&,int)> fn2 = preProcess_peleenet;
			op.dispatch(fn2, buf_id);
		}
	}
	lock_id.unlock();
	return false;
}

bool inference_peleenet(dispatch_queue &op, int buf_id)
{
	std::unique_lock<std::mutex> lock_id(op.lock_inf);
	if(buf_id == -1)
	{
		std::function<bool(dispatch_queue&,int)> fn1 = postProcess_peleenet;
		op.dispatch(fn1, -1);
		op.dead_cnt = op.dead_cnt + 1;
		if(op.dead_cnt == MAX_THREADS)
		{
			cout<<"avg time:"<<(total_time/(cnt-1))<<"(ms)"<<endl;
			cout<<"pre avg time:"<<(pre_total/cnt)<<"(ms)"<<endl;
			cout<<"inf avg time:"<<(inf_total/cnt)<<"(ms)"<<endl;
			cout<<"post avg time:"<<(post_total/cnt)<<"(ms)"<<endl;
			op.unlock_main = true;
			op.wait_main.notify_all();
		}
		return true;
	}
	else
	{
		auto in1 = std::chrono::high_resolution_clock::now();
		op.set_input("input.1", op.in_P[buf_id]);
		op.run();
		op.get_output(0, op.out_bd[buf_id]);
		if(flag_detect)
		{
			op.get_output(1,op.out_con[buf_id]);	
		}
		
		auto t1 = std::chrono::high_resolution_clock::now();
		float duration = 0.0f;
		if(cnt)
			duration = std::chrono::duration_cast<std::chrono::microseconds>( t1 -  prev_time).count() / 1000;
		prev_time = t1;
		total_time += duration;
		cnt++;
		
		auto in2 = std::chrono::high_resolution_clock::now();
		auto inf_duration = std::chrono::duration_cast<std::chrono::microseconds>( in2 -  in1).count() / 1000;
		inf_total += inf_duration;
		
		std::function<bool(dispatch_queue&,int)> fn1 = postProcess_peleenet;
		op.dispatch(fn1, buf_id);
	}
	lock_id.unlock();
	return false;
}

//std::chrono::system_clock::time_point prev_time;
bool postProcess_peleenet(dispatch_queue &op, int buf_id)
{
	std::unique_lock<std::mutex> lock_id(op.lock_post);
	if(buf_id == -1)
	{
		std::function<bool(dispatch_queue&,int)> fn1 = preProcess_peleenet;
		op.dispatch(fn1, -1);
		op.dead_cnt = op.dead_cnt + 1;
		if(op.dead_cnt == MAX_THREADS)
		{
			cout<<"avg time:"<<(total_time/(cnt-1))<<"(ms)"<<endl;
			cout<<"pre avg time:"<<(pre_total/cnt)<<"(ms)"<<endl;
			cout<<"inf avg time:"<<(inf_total/cnt)<<"(ms)"<<endl;
			cout<<"post avg time:"<<(post_total/cnt)<<"(ms)"<<endl;
			op.unlock_main = true;
			op.wait_main.notify_all();
		}
		return true;
	}
	else
	{
		auto po1 = std::chrono::high_resolution_clock::now();
		if(flag_detect)
		{
			decode_boxes(static_cast<float*>(op.out_bd[buf_id]->data),p_boxes,variance, 23280, op.sizes[buf_id][0],op.sizes[buf_id][1]);
    			op.out_put[buf_id].clear();
			//nms(static_cast<float*>(op.out_bd[buf_id]->data), static_cast<float*>(op.out_con[buf_id]->data), op.out_put[buf_id], 23280, 21, 200, 0.3);
			op.sizes[buf_id].clear();
			//std::function<bool(dispatch_queue&,int)> fn1 = Display_peleenet;
			//op.dispatch(fn1, buf_id);
		}
		else
		{
			vector<vector<float>> class_id;
			get_class(static_cast<float*>(op.out_bd[buf_id]->data), class_id);
			for(int i=0; i<class_id.size(); i++)
			{
				cout<<"image no:"<<i<<endl;
				cout<<"class ID:"<<class_id[i][0]<<endl;
				cout<<"class score:"<<class_id[i][1]<<endl;
			}
		}
		/*auto t1 = std::chrono::high_resolution_clock::now();
		float duration = 0.0f;
		if(cnt)
			duration = std::chrono::duration_cast<std::chrono::microseconds>( t1 -  prev_time).count() / 1000;
		prev_time = t1;
		total_time += duration;
		cnt++;*/
		auto po2 = std::chrono::high_resolution_clock::now();
		auto po_duration = std::chrono::duration_cast<std::chrono::microseconds>( po2 -  po1).count() / 1000;
		post_total += po_duration;
		
    		free_buff.push(buf_id);
    		op.app_input.notify_all();
    		op.wait_producer.notify_all();
	}
	lock_id.unlock();
	return false;
}
bool Display_peleenet(dispatch_queue &op, int buf_id)
{
	std::unique_lock<std::mutex> lock_id(op.lock_display);
	if(buf_id == -1)
	{
		std::function<bool(dispatch_queue&,int)> fn2 = preProcess_peleenet;
		op.dispatch(fn2, -1);
		op.dead_cnt = op.dead_cnt + 1;
		if(op.dead_cnt == MAX_THREADS)
		{
			cout<<"avg time:"<<(total_time/cnt)<<"(ms)"<<endl;
			op.unlock_main = true;
			op.wait_main.notify_all();
		}
		return true;
	}
	else
	{
		auto t1 = std::chrono::high_resolution_clock::now();
		float duration = 0.0f;
		if(cnt)
			duration = std::chrono::duration_cast<std::chrono::microseconds>( t1 -  prev_time).count() / 1000;
		prev_time = t1;
		total_time += duration;
		cnt++;
		for (std::map<string, cv::Mat>::iterator it=op.disp_images[buf_id].begin(); it!=op.disp_images[buf_id].end(); ++it)
		{
			for(int bb=0; bb<op.out_put[buf_id].size(); bb++)
    			{
    				int x1,y1,x2,y2;
    				x1 = op.out_put[buf_id][bb][0];
    				y1 = op.out_put[buf_id][bb][1];
    				x2 = op.out_put[buf_id][bb][2];
    				y2 = op.out_put[buf_id][bb][3];
    				rectangle(it->second, Point(x1,y1), Point(x2,y2), cv::Scalar(0, 255, 0));
    			}
    			imshow( "Display window", it->second );                   // Show our image inside it.
    			//imwrite(it->first, it->second);
    			waitKey(0);
    		}
    		free_buff.push(buf_id);
    		op.app_input.notify_all();
    		op.wait_producer.notify_all();
    		if(MAX_THREADS == 1)
    		{
    			std::function<bool(dispatch_queue&,int)> fn2 = preProcess_peleenet;
			op.dispatch(fn2, buf_id);
		}
    	}
    	lock_id.unlock();
    	return false;
}
