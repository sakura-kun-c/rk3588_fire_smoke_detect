#ifndef RK3588_ENGINE_HELPER_H
#define RK3588_ENGINE_HELPER_H

#include <algorithm>
#include <fstream>
#include <string.h>
#include <vector>
#include <rknn_api.h>
#include <cctype>
#include <utility>
#include "utils/logging.h"
#include "utils/datatype.h"

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr)
    {
        NN_LOG_ERROR("fopen %s fail!", filename);
        return nullptr;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char *)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(model, 1, model_len, fp))
    {
        NN_LOG_ERROR("fread %s fail!", filename);
        free(model);
        return nullptr;
    }
    *model_size = model_len;
    if (fp)
    {
        fclose(fp);
    }
    return model;
}

static void print_tensor_attr(rknn_tensor_attr *attr)
{
    NN_LOG_INFO("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
                "zp=%d, scale=%f",
                attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
                attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
                get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

// Helper function: Trim leading and trailing spaces from a string
static std::string trim(const std::string& str) 
{
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

// Helper function: Convert a string to uppercase
static std::string to_upper(const std::string& str)
{
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}



static std::vector<StreamInfo> parse_stream_file(const std::string& stream_path) 
{
    std::ifstream stream_file(stream_path);
    std::vector<StreamInfo> stream_infos;
    std::string line;

    while (std::getline(stream_file, line)) 
    {
        line = trim(line);
        if (line.empty()) continue; // 跳过空行

        std::istringstream iss(line);
        std::string url, model_address, model_type_str, flag_str;

        // 逐个读取非空字段
        if (!(iss >> url >> model_address >> model_type_str >> flag_str)) 
        {
            continue; // 跳过格式错误的行
        }

        model_type_str = trim(model_type_str);
        flag_str = trim(flag_str);

        // 解析 model_type 为整数
        int model_type;
        try 
        {
            model_type = std::stoi(model_type_str); // 尝试转换为整数
        } 
        catch (const std::exception&) 
        {
            std::cerr << "Invalid model type: " << model_type_str << " | use defaule model type 0:" << std::endl;
            model_type = 0;
        }

        // 解析推流标志，大小写不敏感
        bool push_flag = (flag_str == "T" || flag_str == "t");

        stream_infos.push_back({url, model_address, model_type, push_flag});
    }

    return stream_infos;
}


#endif // RK3588_ENGINE_HELPER_H
