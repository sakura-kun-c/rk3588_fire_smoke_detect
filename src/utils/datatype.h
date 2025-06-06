#ifndef zl_mpp_datatpe_h_
#define zl_mpp_datatpe_h_

#include <stdio.h>
#include <opencv2/opencv.hpp>

#include "rknn_api.h"

struct StreamInfo {
    std::string url;
    std::string model_address;              // 模型地址
    int model_type;                         // "0-DET", "1-SEG", "2-POSE", "3-INCLUDE"
    bool push_flag;                         // true if 'T', false if 'F'
};


typedef struct
{ 
    rknn_tensor_attr attr;
    void *data;
} rknn_tensor_data_c;                       // custom struct

typedef struct _nn_object_s 
{
    float x;
    float y;
    float w;
    float h;
    float score;
    int class_id;
} nn_object_s;

struct Detection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
};

#endif // zl_mpp_datatpe_h_