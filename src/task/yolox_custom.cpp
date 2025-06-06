#include <random>
#include "yolox_custom.h"
#include "logging.h"
// #include <chrono>
// #include <iostream>
// #include <sstream>

static std::vector<std::string> g_classes = {"fire", "smoke"};

YoloxCustom::YoloxCustom()
{
    rknn_engine_ = CreateRKNNEngine();
    input_tensor_.data = nullptr;
    want_float_ = false;
}

YoloxCustom::~YoloxCustom()
{
    // release input tensor and output tensor
    NN_LOG_DEBUG("release input tensor");
    if (input_tensor_ .data!= nullptr)
    {
        free(input_tensor_.data);
        input_tensor_.data = nullptr;
    }
    NN_LOG_DEBUG("release output tensor");
    for (auto &tensor : output_tensors_)
    {
        if (tensor.data != nullptr)
        {
            free(tensor.data);
            tensor.data = nullptr;
        }
    }    
}

int YoloxCustom::LoadModel(const char* model_file_path)
{
    auto ret = rknn_engine_->LoadModelFile(model_file_path);
    if (ret != 0)
    {
        NN_LOG_ERROR("yolov8 load model file failed");
        return ret;
    }
    
    auto input_attrs = rknn_engine_->GetInputAttrs();
    if (input_attrs.size() != 1)
    {
        NN_LOG_ERROR("yolov8 input tensor number is not 1, but %ld", input_attrs.size());
        return -1;
    }

    input_tensor_.attr.n_dims = input_attrs[0].n_dims;
    input_tensor_.attr.index = 0;
    input_tensor_.attr.type = RKNN_TENSOR_UINT8;
    input_tensor_.attr.fmt = RKNN_TENSOR_NHWC;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        input_tensor_.attr.dims[0] = input_attrs[0].dims[0];
        input_tensor_.attr.dims[1] = input_attrs[0].dims[2];
        input_tensor_.attr.dims[2] = input_attrs[0].dims[3];
        input_tensor_.attr.dims[3] = input_attrs[0].dims[1];
    }
    else if (input_attrs[0].fmt == RKNN_TENSOR_NHWC)
    {
        input_tensor_.attr.dims[0] = input_attrs[0].dims[0];
        input_tensor_.attr.dims[1] = input_attrs[0].dims[1];
        input_tensor_.attr.dims[2] = input_attrs[0].dims[2];
        input_tensor_.attr.dims[3] = input_attrs[0].dims[3];
    }
    else
    {
        NN_LOG_ERROR("unsupported input layout");
        exit(-1);
    }
    
    input_tensor_.attr.n_elems = input_attrs[0].dims[0] * input_attrs[0].dims[1] * 
                                 input_attrs[0].dims[2] * input_attrs[0].dims[3];

    input_tensor_.attr.size = input_tensor_.attr.n_elems * sizeof(uint8_t);
    input_tensor_.data = malloc(input_tensor_.attr.size);

    auto output_attrs = rknn_engine_->GetOuputAttrs();
    if (output_attrs[0].type == RKNN_TENSOR_FLOAT16)
    {
        want_float_ = true;
        NN_LOG_WARNING("yolov8 output tensor type is float16, want type set to float32");
    }

    for (int i = 0; i < output_attrs.size(); i++)
    {
        rknn_tensor_data_c tensor;

        tensor.attr.n_elems = output_attrs[i].n_elems;
        tensor.attr.n_dims = output_attrs[i].n_dims;
        for (int j = 0; j < output_attrs[i].n_dims; j++)
        {
            tensor.attr.dims[j] = output_attrs[i].dims[j];
        }
        // output tensor needs to be float32
        tensor.attr.type = want_float_ ? RKNN_TENSOR_FLOAT16 : output_attrs[i].type;
        tensor.attr.index = 0;
        tensor.attr.size = output_attrs[i].n_elems * sizeof(get_type_string(tensor.attr.type));
        tensor.data = malloc(tensor.attr.size);
        output_tensors_.push_back(tensor);
        out_zps_.push_back(output_attrs[i].zp);
        out_scales_.push_back(output_attrs[i].scale);  
    }
    
    return 0;
}

int YoloxCustom::Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox)
{
    float wh_ratio = (float)input_tensor_.attr.dims[2] / (float)input_tensor_.attr.dims[1];
    // lettorbox
    if (process_type == "opencv")
    {
        // BGR2RGB，resize，再放入input_tensor_中
        letter_box_info_ = letterbox(img, image_letterbox, wh_ratio);
        letter_box_info_.scale_ratio = float(image_letterbox.cols) /  float(input_tensor_.attr.dims[2]);
        cvimg2tensor(image_letterbox, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);
    }
    return 0;
}

int YoloxCustom::Inference()
{
    std::vector<rknn_tensor_data_c> inputs;
    inputs.push_back(input_tensor_);
    return rknn_engine_->Inference(inputs, output_tensors_, want_float_);
}

int YoloxCustom::Postprocess(const cv::Mat &img, std::vector<Detection> &objects)
{
    void *output_data[output_tensors_.size()];
    NN_LOG_DEBUG("output_tensors_.size():%d", output_tensors_.size());
    for (int i = 0; i < output_tensors_.size(); i++)
    {
        output_data[i] = (void *)output_tensors_[i].data;
    }
    
    std::vector<float> DetectiontRects;

    if (want_float_)
    {
        forlinx::GetResult((float **)output_data, DetectiontRects);
    }
    else
    {
        forlinx::GetResultInt8((int8_t **)output_data, out_zps_, out_scales_, DetectiontRects);
    }
    
    int img_width = img.cols;
    int img_height = img.rows;
    for (int i = 0; i < DetectiontRects.size(); i += 6)
    {
        int classId = int(DetectiontRects[i + 0]);
        float conf = DetectiontRects[i + 1];
        int xmin = int(DetectiontRects[i + 2] * letter_box_info_.scale_ratio);
        int ymin = int(DetectiontRects[i + 3] * letter_box_info_.scale_ratio);
        int xmax = int(DetectiontRects[i + 4] * letter_box_info_.scale_ratio);
        int ymax = int(DetectiontRects[i + 5] * letter_box_info_.scale_ratio);
        
        Detection result;
        result.class_id = classId;
        result.confidence = conf;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen),
                                  dis(gen),
                                  dis(gen));

        result.className = g_classes[result.class_id];
        result.box = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);

        objects.push_back(result);
    }

    return 0;
}

void letterbox_decode(std::vector<Detection> &objects, bool hor, int pad)
{
    for (auto &obj : objects)
    {
        if (hor)
        {
            obj.box.x -= pad;
        }
        else
        {
            obj.box.y -= pad;
        }
    }
}


int YoloxCustom::Run(const cv::Mat &img, std::vector<Detection> &objects)
{
    auto start = std::chrono::steady_clock::now();
    
    cv::Mat image_letterbox;
    Preprocess(img, "opencv", image_letterbox);
    auto preprocess_end = std::chrono::steady_clock::now();

    Inference();
    auto inference_end = std::chrono::steady_clock::now();

    Postprocess(image_letterbox, objects);
    auto postprocess_end = std::chrono::steady_clock::now();

    letterbox_decode(objects, letter_box_info_.hor, letter_box_info_.pad);

    auto preprocess_time = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - start).count();
    auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - preprocess_end).count();
    auto postprocess_time = std::chrono::duration_cast<std::chrono::milliseconds>(postprocess_end - inference_end).count();
    std::ostringstream log;
    log << "Preprocess time: " << preprocess_time << " ms, "
        << "Inference time: " << inference_time << " ms, "
        << "Postprocess time: " << postprocess_time << " ms," << "ms\n";
    
    std::cout << log.str();

    return 0;    
}

