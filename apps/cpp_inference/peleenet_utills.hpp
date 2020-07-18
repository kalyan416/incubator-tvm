#ifndef __peleenet_utills_hpp__
#define __peleenet_utills_hpp__
#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<dirent.h>
#include<map>
#include <assert.h>
#include<math.h>

using namespace std;
using namespace cv;
void classifer_pre_process(float* in_ptr, cv::Mat image, float mean[3], float sd[3], int batch_size=1, int resize=224);
void classifer_pre_process(float* in_ptr, std::vector<cv::Mat> &images, float mean[3], float sd[3], int batch_size=1, int resize=224);
void peleeNet_pre_pocess(float* in_ptr, cv::Mat image, float mean[3], std::vector<int> &sizes, bool rect=false,int batch_size=1, int resize=304);
void peleeNet_pre_pocess(float* in_ptr, std::vector<cv::Mat> &images, float mean[3], std::vector<int> &sizes, bool rect=false, int batch_size=1, int resize=304);
 std::map<string, cv::Mat> peleeNet_pre_pocess(float* in_ptr, std::string imgs_path, float mean[3], std::vector<int> &sizes, bool rect=false, int batch_size=1, int resize=304);
 std::map<string, cv::Mat> classifer_pre_process(float* in_ptr, std::string imgs_path, float mean[3], float std[3], int batch_size=1, int resize=224);
 void get_class(float *out_put, std::vector<std::vector<float>> &pred_cls_id_conf, int num_classes=1000, int batch_size=1);
 void get_min_max_sizes(int min_ratio, int max_ratio, int input_size, int num_feature_maps, std::vector<std::vector<float>> &out_sizes);
 float* priorbox_generation(int input_size, std::vector<int> feature_maps, std::vector<int> steps, std::vector<std::vector<float>> min_max_sizes, std::vector<float> aspect_ratios, unsigned int num_boxes, bool rect=false, bool clip=true);
 void decode_boxes(float *loc_del, float* prior_boxes, float variances[2], unsigned int num_boxes, int width, int height, int batch_size=1);
 void nms(float* boxes, float* conf, std::vector<std::vector<float>> &out_boxes, unsigned int num_boxes, int num_cls,int  num_p_cls,float iou_th = 0.3, float threshold=0.001, int batch_size=1, float score_th=0.5);
 #endif
