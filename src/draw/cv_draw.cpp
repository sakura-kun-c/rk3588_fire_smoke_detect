#include "cv_draw.h"

// draw bbox on img
void DrawDetections(cv::Mat &img, const std::vector<Detection> &objects)
{
    int img_width = img.cols;
    int img_height = img.rows;

    float fontScale = std::max(0.5f, std::min(img_width, img_height) / 640.0f); 
    int thickness = std::max(1, (int)(std::min(img_width, img_height) / 400.0f));

    for (const auto &object : objects)
    {
        cv::Scalar boxColor;
        if (object.className == "fire")
            boxColor = cv::Scalar(255, 0, 0);
        else if (object.className == "smoke")
            boxColor = cv::Scalar(0, 255, 0); 
        else
            boxColor = object.color;

        cv::rectangle(img, object.box, boxColor, thickness);

        std::ostringstream label_ss;
        label_ss << object.className << " " << std::fixed << std::setprecision(2) << object.confidence;
        std::string label = label_ss.str();

        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
        int top = std::max(object.box.y, label_size.height);

        cv::rectangle(img, cv::Point(object.box.x, top - label_size.height - 5),
                      cv::Point(object.box.x + label_size.width, top + baseline - 5),
                      boxColor, cv::FILLED);

        cv::putText(img, label, cv::Point(object.box.x, top - 5),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thickness);
    }
}







