#include"peleenet_utills.hpp"

static DIR *dir, *dir1;
static struct dirent *ent, *ent1;
static bool in_it = false, in_it1=false;
static int img_size, pl_size, row_size, img_size1, pl_size1, row_size1;
void get_class(float *out_put, std::vector<std::vector<float>> &pred_cls_id_conf, int num_classes, int batch_size)
{
	float tmp_sr ;
	int tmp_id = -1;
	for(int i=0; i<batch_size; i++)
	{
		tmp_sr = 0.0f;
		std::vector<float> t_;
		for(int j=0; j<num_classes; j++)
		{
			if(out_put[i*num_classes + j] > tmp_sr)
			{
				tmp_sr = out_put[i*num_classes + j];
				tmp_id = j;
			}
		}
		cout<<tmp_sr<<"   "<<tmp_id<<endl;
		t_.push_back(tmp_id);
		t_.push_back(tmp_sr);
		pred_cls_id_conf.push_back(t_);
	}
}
std::map<string, cv::Mat> classifer_pre_process(float* in_ptr, std::string imgs_path, float mean[3], float sd[3], int batch_size, int resize)
{
	std::map<string, cv::Mat> NameToMat;
 	std::string img_path, name;
 	cv::Mat img;
 	unsigned char *im_data;
 	int ch;
 	if(!in_it1)
 	{
 		dir1 = opendir(imgs_path.c_str());
 		in_it1 = true;
 		pl_size1 = resize * resize;
 		img_size1 = pl_size1 * 3;
 		row_size1 = (resize+32) * 3;
 	}
 	for(int i=0; i<batch_size; i++)
 	{
 		if((ent1 = readdir(dir1)) == NULL)
 		{
 			closedir(dir1);
 			break;
 		}
 		if(ent1->d_name[0] == '.')
 		{
 			i--;
 			continue;
 		}
 		name = ent1->d_name;
 		img_path = imgs_path + name;
 		img = cv::imread(img_path.c_str(), 1);
 		if(img.data==NULL)
 		{
    			cout<<"image read failed "<<endl;
    			printf("%s\n",img_path.c_str());
    			exit(1) ;
  		}
  		cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
  		int int_width, int_height;
  		int_width = resize + 32;
  		int_height = int_width;
  		NameToMat[name] = img.clone();
  		if((img.cols <= img.rows and img.cols == int_width) or (img.rows <= img.cols and img.rows == int_width))
  		{
  		
  		}
  		else
  		{
  			if(img.cols < img.rows)
  			{
  				int_height *=  (img.rows / img.cols);
  			}
  			else
  			{
  				int_width *= (img.cols / img.rows);
  			}
  			cv::resize(img, img, cv::Size(int_width,int_height), 0, 0, cv::INTER_LINEAR);
  		}
  		int crop_top = (int_height - resize) / 2.0f + 0.5f;
  		int crop_left = (int_width - resize) / 2.0f + 0.5f;
  		//cv :: Rect roi(crop_top, crop_left, resize, resize);
  		//cv :: Mat crop = img(roi);
  		//im_data = crop.data;
  		im_data = img.data;
  		ch = img.channels();
  		for(int j=0; j<resize; j++)
  		{
  			for(int k=0; k<resize; k++)
  			{
  				for(int l=0; l<3; l++)
  				{
  					in_ptr[i * img_size1 + l *  pl_size1 + j * resize + k] = (((float)im_data[(j+16) * row_size1 + (k+16) * ch + l])/255 - mean[l]) / sd[l];
  					//in_ptr[i * img_size1 + l *  pl_size1 + j * resize + k] = ((float)im_data[(j+16) * (256*3) + (k+16) * ch + l])/255;;
  				}
  			}
  		}
  	}
 	return NameToMat;
}
void classifer_pre_process(float* in_ptr, cv::Mat image, float mean[3], float sd[3], int batch_size, int resize)
{
	std::vector<cv::Mat> images;
	images.push_back(image);
	classifer_pre_process(in_ptr, images, mean, sd, batch_size, resize);
}
void classifer_pre_process(float* in_ptr, std::vector<cv::Mat> &images, float mean[3], float sd[3], int batch_size, int resize)
{
 	cv::Mat img;
 	unsigned char *im_data;
 	int ch;
 	if(!in_it)
 	{
 		in_it = true;
 		pl_size1 = resize * resize;
 		img_size1 = pl_size1 * 3;
 		row_size1 = (resize + 32) * 3;
 	}
 	for(int i=0; i<batch_size; i++)
 	{
 		img = images[i];
  		cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
  		int int_width, int_height;
  		int_width = resize + 32;
  		int_height = int_width;
  		if((img.cols <= img.rows and img.cols == int_width) or (img.rows <= img.cols and img.rows == int_width))
  		{
  		
  		}
  		else
  		{
  			if(img.cols < img.rows)
  			{
  				int_height *=  (img.rows / img.cols);
  			}
  			else
  			{
  				int_width *= (img.cols / img.rows);
  			}
  			cv::resize(img, img, cv::Size(int_width,int_height), 0, 0, cv::INTER_LINEAR);
  		}
  		int crop_top = (int_height - resize) / 2.0f + 0.5f;
  		int crop_left = (int_width - resize) / 2.0f + 0.5f;
  		//cv :: Rect roi(crop_top, crop_left, resize, resize);
  		//cv :: Mat crop = img(roi);
  		//im_data = crop.data;
  		im_data = img.data;
  		ch = img.channels();
  		for(int j=0; j<resize; j++)
  		{
  			for(int k=0; k<resize; k++)
  			{
  				for(int l=0; l<3; l++)
  				{
  					in_ptr[i * img_size1 + l *  pl_size1 + j * resize + k] = (((float)im_data[(j+16) * row_size1 + (k+16) * ch + l])/255 - mean[l]) / sd[l];
  					//in_ptr[i * img_size1 + l *  pl_size1 + j * resize + k] = ((float)im_data[(j+16) * (256*3) + (k+16) * ch + l])/255;;
  				}
  			}
  		}
  	}
 	return ;
}
void peleeNet_pre_pocess(float* in_ptr, cv::Mat image, float mean[3], std::vector<int> &sizes, bool rect, int batch_size, int resize)
{
	std::vector<cv::Mat> images;
	images.push_back(image);
	peleeNet_pre_pocess(in_ptr, images, mean, sizes, rect, batch_size, resize);
}
void peleeNet_pre_pocess(float* in_ptr, std::vector<cv::Mat> &images, float mean[3], std::vector<int> &sizes, bool rect, int batch_size, int resize)
 {
 	cv::Mat img;
 	unsigned char *im_data;
 	int ch;
 	unsigned int width, height;
 	if(rect)
 	{
 		width = 2 * resize;
 		height = resize;
 	}
 	else
 	{
 		width = resize;
 		height = resize;
 	}
 	if(!in_it)
 	{
 		in_it = true;
 		pl_size = width * height;
 		img_size = pl_size * 3;
 		row_size = width * 3;
 	}
 	
 	for(int i=0; i<batch_size; i++)
 	{
 		img = images[i];
 		sizes.push_back(img.cols);
 		sizes.push_back(img.rows);
 		cv::resize(img, img, cv::Size(width, height));
 		ch = img.channels();
 		im_data = img.data;
 		for(int j=0; j<height; j++)
 			for(int k=0; k<width; k++)
 				for(int l=0; l<ch; l++)
 				{
 					in_ptr[(i * img_size) + l * pl_size + j * width + k] = ((float)im_data[j * row_size + k * ch + l]) - mean[l];
 				}
 	}
 	return ;
 }
 std::map<string, cv::Mat> peleeNet_pre_pocess(float* in_ptr, std::string imgs_path, float mean[3], std::vector<int> &sizes, bool rect, int batch_size, int resize)
 {
 	std::map<string, cv::Mat> NameToMat;
 	std::string img_path, name;
 	cv::Mat img;
 	unsigned char *im_data;
 	int ch;
 	unsigned int width, height;
 	if(rect)
 	{
 		width = 2 * resize;
 		height = resize;
 	}
 	else
 	{
 		width = resize;
 		height = resize;
 	}
 	if(!in_it)
 	{
 		dir = opendir(imgs_path.c_str());
 		in_it = true;
 		pl_size = width * height;
 		img_size = pl_size * 3;
 		row_size = width * 3;
 	}
 	
 	for(int i=0; i<batch_size; i++)
 	{
 		if((ent = readdir(dir)) == NULL)
 		{
 			closedir(dir);
 			break;
 		}
 		if(ent->d_name[0] == '.')
 		{
 			i--;
 			continue;
 		}
 		name = ent->d_name;
 		img_path = imgs_path + name;
 		img = cv::imread(img_path.c_str(), 1);
 		if(img.data==NULL)
 		{
    			cout<<"image read failed "<<endl;
    			printf("%s\n",img_path.c_str());
    			exit(1) ;
  		}
 		sizes.push_back(img.cols);
 		sizes.push_back(img.rows);
 		NameToMat[name] = img.clone();
 		cv::resize(img, img, cv::Size(width, height));
 		ch = img.channels();
 		im_data = img.data;
 		for(int j=0; j<height; j++)
 			for(int k=0; k<width; k++)
 				for(int l=0; l<ch; l++)
 				{
 					in_ptr[(i * img_size) + l * pl_size + j * width + k] = ((float)im_data[j * row_size + k * ch + l]) - mean[l];
 				}
 	}
 	return NameToMat;
 }
void get_min_max_sizes(int min_ratio, int max_ratio, int input_size, int num_feature_maps, std::vector<std::vector<float>> &out_sizes)
 {
 	float step = (max_ratio - min_ratio)/(num_feature_maps - 2);
 	float max_p = max_ratio + step + 1;
 	std::vector<float> min_sizes, max_sizes;
 	if(min_ratio == 20)
 	{
 		min_sizes.push_back(input_size * 0.1f); //            10/100
 		max_sizes.push_back(input_size * 0.2f);    //         20/100
 	}
 	else
 	{
 		min_sizes.push_back(input_size * 0.07f); //            7/100
 		max_sizes.push_back(input_size * 0.15f); // 	    15/100
 	}
 	for(float i=min_ratio+step ; i<max_p; i=i+step)
 	{
 		max_sizes.push_back(input_size * i/100);
 	}
 	min_sizes.insert(min_sizes.end(), max_sizes.begin(), --max_sizes.end());
 	out_sizes.push_back(min_sizes);
 	out_sizes.push_back(max_sizes);
 }
float* priorbox_generation(int input_size, std::vector<int> feature_maps, std::vector<int> steps, std::vector<std::vector<float>> min_max_sizes, std::vector<float> aspect_ratios, unsigned int num_boxes, bool rect, bool clip)
 {
 	cout<<feature_maps.size()<<" "<<min_max_sizes[0].size()<<" "<<min_max_sizes[1].size()<<endl;
 	assert(feature_maps.size() == steps.size());
 	assert(min_max_sizes.size() == 2);
 	assert(feature_maps.size() == min_max_sizes[0].size());
 	assert(feature_maps.size() == min_max_sizes[1].size());
 	float* prior_boxes = new float[num_boxes * 4 * sizeof(float)];
 	unsigned int num_f = feature_maps.size();
 	int num_ar = aspect_ratios.size();
 	std::vector<float> tmp(2);
 	std::vector<std::vector<std::vector<float>>> op_ar(num_f);
 	for(int i=0; i<num_f; i++)
 	{
 		for(int a=0; a<num_ar ; a++)
 		{
 			tmp[0] = sqrt(aspect_ratios[a]) / input_size;
 			tmp[1] = sqrt(aspect_ratios[a]) * input_size;
 			tmp[1] = 1 / tmp[1] ;
 			op_ar[i].push_back(tmp);
 		}
 	}
 	unsigned int offset = 0;
 	for(int i=0; i<num_f; i++)
 	{
 		unsigned int f_size_h = feature_maps[i];
 		unsigned int f_size_w;
 		if(rect)
 		{
 			f_size_w = f_size_h;
 		}
 		else
 		{
 			f_size_w = 2 * f_size_h;
 		}
 		float f_k, s_k, s_k_prime, cx, cy;
 		f_k =  steps[i] / float(input_size);
 		s_k = min_max_sizes[0][i] / input_size;
 		s_k_prime = s_k * (min_max_sizes[1][i] / input_size);
 		s_k_prime = sqrt(s_k_prime);
 		if(clip)
 		{
 			s_k = std::min(s_k, 1.0f);
 			s_k_prime = std::min(s_k_prime, 1.0f);
 		}
 		for(int a=0; a<num_ar ; a++)
 		{
 			op_ar[i][a] [0] = min_max_sizes[0][i] * op_ar[i][a] [0];
 			op_ar[i][a] [1] = min_max_sizes[0][i] * op_ar[i][a] [1];
 			if(clip)
 			{
 				op_ar[i][a] [0] = std::min(op_ar[i][a] [0], 1.0f);
 				op_ar[i][a] [1] = std::min(op_ar[i][a] [1], 1.0f);
 			}
 		}
 		
 		for(int j=0; j<f_size_h; j++)
 		{
 			cy = (j + 0.5) * f_k;
 			if(clip)
 				cy = std::min(cy, 1.0f);
 			for(int k=0; k<f_size_w; k++)
 			{
 				cx = (k + 0.5) * f_k;
 				if(clip)
 					cx = std::min(cx, 1.0f);
 				prior_boxes[offset] = cx;
 				prior_boxes[offset + 1] = cy;
 				prior_boxes[offset + 2] = s_k;
 				prior_boxes[offset + 3] = s_k;
 				//printf("%f %f %f %f\n",cx, cy, s_k, s_k);
 				offset += 4;
 				
 				prior_boxes[offset] = cx;
 				prior_boxes[offset + 1] = cy;
 				prior_boxes[offset + 2] = s_k_prime;
 				prior_boxes[offset + 3] = s_k_prime;
 				//printf("%f %f %f %f\n",cx, cy, s_k_prime, s_k_prime);
 				offset += 4;
 				
 				for(int a=0; a<num_ar; a++)
 				{
 					prior_boxes[offset] = cx;
 					prior_boxes[offset + 1] = cy;
 					prior_boxes[offset + 2] = op_ar[i][a] [0];
 					prior_boxes[offset + 3] = op_ar[i][a] [1];
 					//printf("%f %f %f %f\n",cx, cy, op_ar[i][a] [0], op_ar[i][a] [1]);
 					offset += 4;
 					
 					prior_boxes[offset] = cx;
 					prior_boxes[offset + 1] = cy;
 					prior_boxes[offset + 2] = op_ar[i][a] [1];
 					prior_boxes[offset + 3] = op_ar[i][a] [0];
 					//printf("%f %f %f %f\n",cx, cy, op_ar[i][a] [1], op_ar[i][a] [0]);
 					offset += 4;
 				}
 			}
 		}
 	}
 	return prior_boxes;
 }
void decode_boxes(float *loc_del, float* prior_boxes, float variances[2], unsigned int num_boxes, int width, int height, int batch_size)
{
 	unsigned int offset1 = 0, offset2;
 	for(int i=0; i<batch_size; i++)
	{
		offset2 = 0;
		for(unsigned int j=0; j<num_boxes; j++)
		{
			loc_del[offset1 ] = prior_boxes[offset2] + loc_del[offset1 ] * variances[0] * prior_boxes[offset2 + 2];
			loc_del[offset1 + 1] = prior_boxes[offset2 + 1] + loc_del[offset1 + 1] * variances[0] * prior_boxes[offset2 + 3];
			loc_del[offset1 + 2] = prior_boxes[offset2 + 2] * std::exp(loc_del[offset1 + 2] * variances[1]);
			loc_del[offset1 + 3] = prior_boxes[offset2 + 3] * std::exp(loc_del[offset1 + 3] * variances[1]);
			offset2 += 4;
			
			loc_del[offset1 ] = (loc_del[offset1 ] - loc_del[offset1 + 2] * 0.5) ;
			loc_del[offset1 + 1] = (loc_del[offset1 + 1] - loc_del[offset1 + 3] * 0.5) ;
			loc_del[offset1 + 2] = (loc_del[offset1 + 2] + loc_del[offset1 ]) ;
			loc_del[offset1 + 3] = (loc_del[offset1 + 3] + loc_del[offset1 + 1]);
			
			loc_del[offset1 ] *= width;
			loc_del[offset1 + 1] *= height;
			loc_del[offset1 + 2] *= width;
			loc_del[offset1 + 3] *= height;
			offset1 += 4;
		}
	}
}
void fill_indices(std::vector<std::vector<int>> &indices, float *conf, unsigned int num_boxes, int num_cls)
{
	for(int i=0; i<num_cls; i++)
	{
		for(int j=0; j<num_boxes; j++)
		{
			if(conf[j*num_cls+i] > 0.01)
				indices[i].push_back(j);
		}
	}
}
void nms(float* boxes, float* conf, std::vector<std::vector<float>> &out_boxes, unsigned int num_boxes, int num_cls,int  num_p_cls,float iou_th, float threshold, int batch_size, float score_th)
{
	unsigned int offset1, offset2, N,  bnum_img, snum_img;
	offset1 = 0;
	offset2 = 0;
	bnum_img = num_boxes * 4;
	snum_img = num_boxes * num_cls;
	//conf = new float[snum_img * batch_size];
	//memcpy(conf, conf_ori, snum_img * batch_size * sizeof(float));
	for(int i=0; i<batch_size; i++)
	{
		offset1 = i * bnum_img;
		offset2 = i * snum_img;
		std::vector<std::vector<int>> indices(num_cls);
		fill_indices(indices, conf + offset2, num_boxes, num_cls);
		for(int j=1; j<num_cls; j++)
		{
			N = indices[j].size();
			if(N==0)
				continue;
			for(int k=0; k<N; k++)
			{
				int max_pos = indices[j] [k];
				int pos = indices[j] [k+1];
				float max_score = conf[offset2 + max_pos*21 + j];
				int iter=k+1;
				int index_hold = k;
				while(iter<N)
				{
					if(conf[offset2 + pos*21 + j] > max_score)
					{
						max_pos = pos;
						max_score = conf[offset2 + pos*21 + j];
						index_hold = iter;
					}
					iter += 1;
					pos = indices[j] [iter];
				}
				//printf("max:%d %f\n",max_pos,max_score);
				float mx1, my1, mx2, my2;
				mx1 = boxes[offset1 + max_pos  * 4 ];
				my1 = boxes[offset1 + max_pos  * 4 + 1];
				mx2 = boxes[offset1 + max_pos  * 4 + 2];
				my2 = boxes[offset1 + max_pos  * 4 + 3];
				/*boxes[offset1 + max_pos  * 4 ] = boxes[offset1 + k  * 4 ];
				boxes[offset1 + max_pos  * 4 + 1] = boxes[offset1 + k  * 4 + 1];
				boxes[offset1 + max_pos  * 4 + 2] = boxes[offset1 + k  * 4 + 2];
				boxes[offset1 + max_pos  * 4 + 3] = boxes[offset1 + k  * 4 + 3];
				
				boxes[offset1 + k  * 4 ] = mx1;
				boxes[offset1 + k  * 4 + 1] = my1;
				boxes[offset1 + k  * 4 + 2] = mx2;
				boxes[offset1 + k  * 4 + 3] = my2;*/
				
				
				indices[j] [index_hold] = indices[j] [k];
				indices[j] [k] = max_pos;
				iter = k+1;
				
				float x1,y1,x2,y2;
				float area, iw, ih, ua, ov, weight;
				while(iter<N)
				{
					pos = indices[j] [iter];
					x1 = boxes[offset1 + pos  * 4 ];
					y1 = boxes[offset1 + pos  * 4 + 1];
					x2 = boxes[offset1 + pos  * 4 + 2];
					y2 = boxes[offset1 + pos  * 4 + 3];
					area = (x2 - x1 + 1) * (y2 - y1 + 1);
					
					iw = (std::min(mx2, x2) - std::max(mx1, x1) + 1);
					if(iw > 0)
					{
						ih = (std::min(my2, y2) - std::max(my1, y1) + 1);
						
						if(ih > 0)
						{
							ua = (mx2 - mx1 + 1) * (my2 - my1 + 1) + area - (iw * ih);
							ov = iw * ih / ua;
							if(ov > iou_th)
							{
								weight = 1 - ov;
							}
							else
							{
								weight = 1;
							}
							conf[offset2 + pos * 21 + j] *= weight ;
							if(conf[offset2 + pos * 21 + j] < threshold)
							{
								indices[j].erase(indices[j].begin()+iter);
								N -= 1;
								iter = iter-1;
							}
						}
					}
					iter = iter + 1;
				}
			}
			std::vector<float> keep_box(7);
			//std::vector<std::vector<std::vector<float>>> keep_boxes(21);
			int nb = std::min((int)indices[j].size(),num_p_cls);
			/*FILE* file;
			char name[10];
			float *loc = new float[nb*5];
			sprintf(name,"%d.y",j);
			file = fopen(name, "r");
			fread(loc, nb*5*sizeof(float), 1, file);
			fclose(file);
			float error = 0.0f;
			int i=0;*/
			for(int val_o=0; val_o < nb; val_o++)
			{
				if(conf[offset2 + indices[j][val_o] * 21 + j] > score_th)
				{
					keep_box[0] = boxes[offset1 + indices[j][val_o] * 4];
					keep_box[1] = boxes[offset1 + indices[j][val_o] * 4 + 1];
					keep_box[2] = boxes[offset1 + indices[j][val_o] * 4 + 2];
					keep_box[3] = boxes[offset1 + indices[j][val_o] * 4 + 3];
					keep_box[4] = float(j);
					keep_box[5] = conf[offset2 + indices[j][val_o] * 21 + j];
					keep_box[6] = float(i);
					/*error += (loc[i++] - keep_box[0]);
					error += (loc[i++] - keep_box[1]);
					error += (loc[i++] - keep_box[2]);
					error += (loc[i++] - keep_box[3]);
					error += (loc[i++] - keep_box[4]);*/
					out_boxes.push_back(keep_box);
				}
			}
			//printf("error:%f\n", error/(nb*5));
		}
	}
}
/*int main()
{
	/*float *ptr = (float*)malloc(1*304*304*3*sizeof(float));
	float *ptr1 = (float*)malloc(1*304*304*3*sizeof(float));
	float a[3] = {103.94, 116.78, 123.68};
	auto emo = peleeNet_pre_pocess(ptr,"/home/kalyan/libraries/Pelee.Pytorch/imgs/VOC/tmp/",a,20);
	FILE* file;
	file = fopen("image1.y", "r");
	fread(ptr1, 304*304*3*sizeof(float), 1, file);
	float error=0.0f;
	for(int i=0; i<304*304*3; i++)
		error += (ptr[i] - ptr1[i]);
	printf("avg error:%f\n",error/(304*304*3));*/
		
	/*std::vector<std::vector<float>> var ;
	 get_min_max_sizes(20,90,304,5,var);
	 
	 std::vector<int> feature_maps{19, 10, 5, 3, 1};
	 std::vector<int> steps{16, 30, 60, 101, 304};
	 std::vector<float> aspect_ratios{2, 3} ;
	 
	 float * p_boxes = priorbox_generation(304, feature_maps, steps, var, aspect_ratios, 2976);
	 float *ptr1 = (float*)malloc(2976*4*sizeof(float));
	 FILE* file;
	file = fopen("prior_boxes.y", "r");
	fread(ptr1, 2976*4*sizeof(float), 1, file);
	float error=0.0f;
	for(int i=0; i<2976*4; i++)
		error += (p_boxes[i] - ptr1[i]);
	printf("avg error:%f\n",error/(2976*4));
	
	float * loc = (float*)malloc(2976*4*sizeof(float));
	float vari[2] = {0.1,0.2};
	//float *ptr1 = (float*)malloc(2976*4*sizeof(float));
	//FILE* file;
	file = fopen("loc4.y", "r");
	fread(loc, 2976*4*sizeof(float), 1, file);
	fclose(file);
	file = fopen("boxes4.y", "r");
	fread(ptr1, 2976*4*sizeof(float), 1, file);
	fclose(file);
	decode_boxes(loc,p_boxes,vari,2976,354,480);
	//float error=0.0f;
	for(int i=0; i<2976*4; i++)
		error += (loc[i] - ptr1[i]);
	printf("avg error:%f\n",error/(2976*4));
	
	FILE* file;
	float * loc = (float*)malloc(2976*4*sizeof(float));
	float * conf = (float*)malloc(2976*21*sizeof(float));
	file = fopen("boxes1.y", "r");
	fread(loc, 2976*4*sizeof(float), 1, file);
	fclose(file);
	file = fopen("scores1.y", "r");
	fread(conf, 2976*21*sizeof(float), 1, file);
	fclose(file);
	std::vector<std::vector<std::vector<float>>> out_boxes;
	nms(loc, conf, out_boxes, 2976, 21, 200, 0.3);
}*/

