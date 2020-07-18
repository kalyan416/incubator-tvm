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

int main(int argc, char** argv)
{
     std::string lib_path, img_path;
     if(argc == 3)
     {
     		lib_path = argv[1];
     		img_path = argv[2];
     }
     else
     {
     		return 0;
     }
     std::string file_path = lib_path + "deploy.so";
    // tvm module for compiled functions
    tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(file_path.c_str());

    file_path = lib_path + "deploy.json";
    // json graph
    std::ifstream json_in(file_path.c_str(), std::ios::in);
    std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();

    file_path = lib_path + "deploy.params";
    // parameters in binary
    std::ifstream params_in(file_path.c_str(), std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();

    // parameters need to be TVMByteArray type to indicate the binary data
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();

    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLGPU;
    int device_id = 0;

    // get global function module for graph runtime
    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);
    
    std::vector<std::vector<float>> var ;
    get_min_max_sizes(20,90,304,5,var);
    std::vector<int> feature_maps{19, 10, 5, 3, 1};
    std::vector<int> steps{16, 30, 60, 101, 304};
    std::vector<float> aspect_ratios{2, 3} ;
	 
    float * p_boxes = priorbox_generation(304, feature_maps, steps, var, aspect_ratios, 2976);

    DLTensor* x;
    int in_ndim = 4;
    int64_t in_shape[4] = {1, 3, 304, 304};
    TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, kDLCPU, device_id, &x);
    float variance[2] = {0.1,0.2};
    // load image data saved in binary
    //std::ifstream data_fin("cat.bin", std::ios::binary);
    //data_fin.read(static_cast<char*>(x->data), 3 * 224 * 224 * 4);
    
    float mean[3] = {103.94, 116.78, 123.68};
    auto t1 = std::chrono::high_resolution_clock::now();
    auto emo = peleeNet_pre_pocess(static_cast<float*>(x->data),img_path, mean);

    // get the function from the module(set input data)
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    set_input("input.1", x);

    // get the function from the module(load patameters)
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);

    // get the function from the module(run it)
    tvm::runtime::PackedFunc run = mod.GetFunction("run");
    run();

    DLTensor* y,* y1;
    int out_ndim = 3;
    int64_t out_shape[3] = {1,2976, 4};
    TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, kDLCPU, device_id, &y);
    out_shape[2] = 21;
    TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, kDLCPU, device_id, &y1);

    // get the function from the module(get output data)
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
    get_output(0, y);
    get_output(1,y1);
    
    decode_boxes(static_cast<float*>(y->data),p_boxes,variance, 2976, 500, 375);
    
    std::vector<std::vector<float>> out_boxes;
    nms(static_cast<float*>(y->data), static_cast<float*>(y1->data), out_boxes, 2976, 21, 200, 0.3);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << duration;
    
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    for (std::map<string, cv::Mat>::iterator it=emo.begin(); it!=emo.end(); ++it)
    {
    	for(int bb=0; bb<out_boxes.size(); bb++)
    	{
    		//for(int ind=0; ind<out_boxes[bb].size(); ind++)
    		{
    			int x1,y1,x2,y2;
    			x1 = out_boxes[bb][0];
    			y1 = out_boxes[bb][1];
    			x2 = out_boxes[bb][2];
    			y2 = out_boxes[bb][3];
    			rectangle(it->second, Point(x1,y1), Point(x2,y2), cv::Scalar(0, 255, 0));
    		}
    	}
    	imshow( "Display window", it->second );                   // Show our image inside it.
    	waitKey(0); 
    }
    //TVMContext ctx{kDLGPU,0};
    //int number=1000;
    //tvm::runtime::PackedFunc time_evaluator = mod.GetFunction("time_evaluator");
    //time_evaluator("run",ctx, 1, number);
    //auto get_micro_time_evaluator = tvm::runtime::WrapTimeEvaluator("run",ctx,number,1);
    // get the maximum position in output vector
    //auto y_iter = static_cast<float*>(y->data);
    //auto max_iter = std::max_element(y_iter, y_iter + 1000);
    //auto max_index = std::distance(y_iter, max_iter);
    //std::cout << "The maximum position in output vector is: " << max_index << std::endl;

    TVMArrayFree(x);
    TVMArrayFree(y);

    return 0;
}
