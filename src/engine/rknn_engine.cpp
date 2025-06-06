#include <memory>
#include <iostream>
#include "rknn_engine.h"
#include "engine_helper.h"
#include "logging.h"
#include <chrono>
static const int g_max_io_num = 10;
static std::vector<std::string> g_classes = {
    "person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
    "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
    "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
    "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush "};

    
int RKNNEngine::LoadModelFile(const char* model_file_path)
{
    int model_len = 0;
    auto model = load_model(model_file_path, &model_len);
    if (model == nullptr)
    {
        NN_LOG_ERROR("load model file %s fail!", model_file_path);
        return -1;        
    }

    int ret = rknn_init(&rknn_ctx_, model, model_len, 0, NULL); // 初始化rknn context
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_init fail! ret=%d", ret);
        return -1;
    }
    NN_LOG_INFO("rknn_init success!");
    is_created = true;
    
    rknn_core_mask core_maks = RKNN_NPU_CORE_0_1_2;
    ret = rknn_set_core_mask(rknn_ctx_, core_maks);    

    rknn_input_output_num io_num;
    ret = rknn_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        NN_LOG_ERROR("rknn_query fail! ret=%d", ret);
        return -1;
    }
    NN_LOG_INFO("model input num: %d, output num: %d", io_num.n_input, io_num.n_output);

    input_num_ = io_num.n_input;
    output_num_ = io_num.n_output;

    NN_LOG_INFO("input tensors:");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            NN_LOG_ERROR("rknn_query fail! ret=%d", ret);
            return -1;
        }
        print_tensor_attr(&(input_attrs[i]));
        // set input_attrs_
        input_attrs_.push_back(input_attrs[i]);
    }
    
    NN_LOG_INFO("output tensors:");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            NN_LOG_ERROR("rknn_query fail! ret=%d", ret);
            return -1;
        }
        print_tensor_attr(&(output_attrs[i]));
        // set ouput_attrs_
        ouput_attrs_.push_back(output_attrs[i]);
    }

    return 0;    

}

int RKNNEngine::Inference(std::vector<rknn_tensor_data_c>& inputs, std::vector<rknn_tensor_data_c>& outputs, bool want_float)
{
    if (inputs.size() != input_num_)
    {
        NN_LOG_ERROR("inputs num not match! inputs.size()=%ld, input_num_=%d", inputs.size(), input_num_);
        return -1;
    }

    if (outputs.size() != output_num_)
    {
        NN_LOG_ERROR("outputs num not match! outputs.size()=%ld, output_num_=%d", outputs.size(), output_num_);
        return -1;
    }

    // rknn_input init
    rknn_input rknn_inputs[g_max_io_num];
    memset(rknn_inputs, 0, sizeof(rknn_inputs));
    rknn_inputs[0].index = inputs[0].attr.index;
    rknn_inputs[0].type = inputs[0].attr.type;
    rknn_inputs[0].size = inputs[0].attr.size;
    rknn_inputs[0].fmt = inputs[0].attr.fmt;
    rknn_inputs[0].buf = inputs[0].data;

    int ret = rknn_inputs_set(rknn_ctx_, (uint32_t)inputs.size(), rknn_inputs);
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_inputs_set fail! ret=%d", ret);
        return -1;
    }
    
    
    ret = rknn_run(rknn_ctx_, nullptr);
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_run fail! ret=%d", ret);
        return -1;
    }


    rknn_output rknn_outputs[g_max_io_num];
    memset(rknn_outputs, 0, sizeof(rknn_outputs));
    for (int i = 0; i < output_num_; ++i)
    {
        rknn_outputs[i].want_float = want_float ? 1 : 0;
    }

    ret = rknn_outputs_get(rknn_ctx_, output_num_, rknn_outputs, NULL);
    if (ret != 0) 
    {
        NN_LOG_ERROR("rknn_outputs_get failed! ret=%d", ret);
        return -1;
    }
    
    for (int i = 0; i < output_num_; ++i)
    {   
        outputs[i].attr.index = rknn_outputs[i].index;
        outputs[i].attr.size = rknn_outputs[i].size;
        // printf("output size: %d \n", rknn_outputs[i].size);
        memcpy(outputs[i].data, rknn_outputs[i].buf, rknn_outputs[i].size);
        free(rknn_outputs[i].buf);
    }
  
    return 0;
}

const std::vector<rknn_tensor_attr>& RKNNEngine::GetInputAttrs() { return input_attrs_; }

const std::vector<rknn_tensor_attr>& RKNNEngine::GetOuputAttrs() { return ouput_attrs_; }

RKNNEngine::~RKNNEngine()
{
    if (is_created)
    {
        rknn_destroy(rknn_ctx_);
        NN_LOG_INFO("rknn context destroyed!");        
    }
}

std::shared_ptr<RKNNEngine> CreateRKNNEngine()
{
    return std::make_shared<RKNNEngine>();
}