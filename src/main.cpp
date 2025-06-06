
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <chrono>

#include "yolox_custom.h"
#include "logging.h"
#include "cv_draw.h"

namespace fs = std::filesystem;
fs::path results_dir = "results";
// 判断是否是图片文件
bool IsImageFile(const fs::path& path) 
{
    static const std::vector<std::string> img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return std::find(img_exts.begin(), img_exts.end(), ext) != img_exts.end();
}

// 判断是否是视频文件
bool IsVideoFile(const fs::path& path) 
{
    static const std::vector<std::string> video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv"};
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return std::find(video_exts.begin(), video_exts.end(), ext) != video_exts.end();
}

// 处理图片目录
void ProcessImagesInDirectory(const std::string& model_file, const std::string& dir_path) 
{
    YoloxCustom yolo;
    yolo.LoadModel(model_file.c_str());

    for (const auto& entry : fs::directory_iterator(dir_path)) 
    {
        if (!entry.is_regular_file()) continue;
        if (!IsImageFile(entry.path())) continue;

        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) 
        {
            NN_LOG_ERROR("Failed to load image: %s", entry.path().string().c_str());
            continue;
        }

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<Detection> objects;
        yolo.Run(img, objects);
        DrawDetections(img, objects);

        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        NN_LOG_INFO("Processed image %s, time: %.2f ms", entry.path().filename().string().c_str(), duration);

        // 保存结果图像
        std::string result_name = entry.path().stem().string() + "_result.jpg";
        fs::path save_path = results_dir / result_name;
        cv::imwrite(save_path.string(), img);
    }
}


// 处理视频目录
void ProcessVideosInDirectory(const std::string& model_file, const std::string& dir_path, bool record = false) 
{
    YoloxCustom yolo;
    yolo.LoadModel(model_file.c_str());

    for (const auto& entry : fs::directory_iterator(dir_path)) 
    {
        if (!entry.is_regular_file()) continue;
        if (!IsVideoFile(entry.path())) continue;

        cv::VideoCapture cap(entry.path().string());
        if (!cap.isOpened()) 
        {
            NN_LOG_ERROR("Failed to open video file: %s", entry.path().string().c_str());
            continue;
        }

        int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        int fps = cap.get(cv::CAP_PROP_FPS);
        NN_LOG_INFO("Processing video %s, size: %d x %d, fps: %d", entry.path().filename().string().c_str(), width, height, fps);
        
        cv::VideoWriter writer;
        if (record) 
        {

            NN_LOG_INFO("save result mp4");
            std::string out_file = (results_dir / (entry.path().stem().string() + "_result.avi")).string();
            // writer = cv::VideoWriter(out_file, cv::VideoWriter::fourcc('M','J','P','G'), fps, cv::Size(width, height));
            writer = cv::VideoWriter(out_file, cv::VideoWriter::fourcc('X','V','I','D'), fps, cv::Size(width, height));

            // std::string out_file = (results_dir / (entry.path().stem().string() + "_result.mp4")).string();
            // writer = cv::VideoWriter(out_file, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));

            if (!writer.isOpened()) 
            {
                NN_LOG_ERROR("Failed to open VideoWriter for file: %s", out_file.c_str());
                continue;
            }

        }

        int frame_count = 0;
        auto start_all = std::chrono::high_resolution_clock::now();

        cv::Mat img;
        while (true) 
        {
            auto start_1 = std::chrono::high_resolution_clock::now();

            cap >> img;
            if (img.empty()) 
            {
                NN_LOG_INFO("Video %s end.", entry.path().filename().string().c_str());
                break;
            }
            std::vector<Detection> objects;
            yolo.Run(img, objects);
            DrawDetections(img, objects);
            if (record) 
            {
                std::cout << "======== img.cols" << img.cols << "======== img.rows" << img.rows << std::endl;
                if (img.channels() != 3 || img.type() != CV_8UC3) 
                {
                    NN_LOG_ERROR("Frame format not supported by VideoWriter: channels=%d, type=%d", img.channels(), img.type());
                    continue;
                }

                writer << img;
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " model_file directory img|video [record]" << std::endl;
        return -1;
    }

    std::string model_file = argv[1];
    std::string dir_path = argv[2];
    std::string mode = argv[3];
    bool record = (argc > 4) ? (atoi(argv[4]) != 0) : false;
    // 创建结果保存目录
    fs::path results_dir = "results";
    if (!fs::exists(results_dir)) 
    {
        fs::create_directory(results_dir);
    }
    if (mode == "img") 
    {
        ProcessImagesInDirectory(model_file, dir_path);
    } 
    else if (mode == "video") 
    {
        ProcessVideosInDirectory(model_file, dir_path, record);
    } 
    else 
    {
        std::cerr << "Invalid mode: " << mode << ". Use 'img' or 'video'." << std::endl;
        return -1;
    }

    return 0;
}