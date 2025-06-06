#ifndef RK3588_YOLOV8_FLOW_H_
#define RK3588_YOLOV8_FLOW_H_

#include <memory>
#include <opencv2/opencv.hpp>
#include "rknn_engine.h"
#include "preprocess.h"
#include "postprocess.h"
#include "utils/datatype.h"

class YoloxCustom
{
public:
    YoloxCustom();
    ~YoloxCustom();

    int LoadModel(const char* model_file_path);
    int Run(const cv::Mat &img, std::vector<Detection> &objects);

private:

    int Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox);
    int Inference();
    int Postprocess(const cv::Mat &img, std::vector<Detection> &objects);

    bool want_float_;
    LetterBoxInfo letter_box_info_;
    rknn_tensor_data_c input_tensor_;
    std::vector<rknn_tensor_data_c> output_tensors_;
    std::shared_ptr<RKNNEngine> rknn_engine_;
    std::vector<int32_t> out_zps_;
    std::vector<float> out_scales_;
};


#endif // RK3588_YOLOV8_FLOW_H_