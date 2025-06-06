#ifndef RKNN_ENGINE_H_
#define RKNN_ENGINE_H_

#include <vector>
#include "rknn_api.h"
#include "datatype.h"


class RKNNEngine
{
public:
    RKNNEngine(): rknn_ctx_(0), input_num_(0), output_num_(0), is_created(false) {};
    ~RKNNEngine();

    int LoadModelFile(const char* model_file_path);
    int Inference(std::vector<rknn_tensor_data_c>& inputs, std::vector<rknn_tensor_data_c>& outputs, bool want_float);
    const std::vector<rknn_tensor_attr>& GetInputAttrs();
    const std::vector<rknn_tensor_attr>& GetOuputAttrs();

private:
    rknn_context rknn_ctx_;
    bool is_created;
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> ouput_attrs_;
    uint32_t input_num_;
    uint32_t output_num_;
        
};
std::shared_ptr<RKNNEngine> CreateRKNNEngine();
#endif // RKNN_ENGINE_H_