#include <string.h>
#include <stdlib.h>
#include <algorithm>

#include "postprocess.h"
#include "logging.h"

namespace forlinx
{
    static int input_w = 640;
    static int input_h = 640;
    static float objectThreshold = 0.45;
    static float nmsThreshold = 0.45;
    static int headNum = 3;
    static int class_num = 2;
    static int strides[3] = {8, 16, 32};
    static int mapSize[3][2] = {{80, 80}, {40, 40}, {20, 20}};         // depends on model input size

#define ZQ_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ZQ_MIN(a, b) ((a) < (b) ? (a) : (b))
    static inline float fast_exp(float x)
    {
        // return exp(x);
        union
        {
            uint32_t i;
            float f;
        } v;
        v.i = (12102203.1616540672 * x + 1064807160.56887296);
        return v.f;
    }

    float sigmoid(float x)
    {
        return 1 / (1 + fast_exp(-x));
    }

    static inline float IOU(float XMin1, float YMin1, float XMax1, float YMax1, float XMin2, float YMin2, float XMax2, float YMax2)
    {
        float Inter = 0;
        float Total = 0;
        float XMin = 0;
        float YMin = 0;
        float XMax = 0;
        float YMax = 0;
        float Area1 = 0;
        float Area2 = 0;
        float InterWidth = 0;
        float InterHeight = 0;

        XMin = ZQ_MAX(XMin1, XMin2);
        YMin = ZQ_MAX(YMin1, YMin2);
        XMax = ZQ_MIN(XMax1, XMax2);
        YMax = ZQ_MIN(YMax1, YMax2);

        InterWidth = XMax - XMin;
        InterHeight = YMax - YMin;

        InterWidth = (InterWidth >= 0) ? InterWidth : 0;
        InterHeight = (InterHeight >= 0) ? InterHeight : 0;

        Inter = InterWidth * InterHeight;

        Area1 = (XMax1 - XMin1) * (YMax1 - YMin1);
        Area2 = (XMax2 - XMin2) * (YMax2 - YMin2);

        Total = Area1 + Area2 - Inter;

        return float(Inter) / float(Total);
    }
    
    inline static int32_t __clip(float val, float min, float max)
    {
        float f = val <= min ? min : (val >= max ? max : val);
        return f;
    }

    static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
    {
        float dst_val = (f32 / scale) + zp;
        int8_t res = (int8_t)__clip(dst_val, -128, 127);
        return res;
    }

    static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

    // int 
    int GetResultInt8(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects)
    { 
        int ret = 0;
        int grid_h, grid_w;
        int model_class_num = class_num;
        std::vector<nn_object_s> detectRects;
        int box_zp = 0, score_zp = 0, score_sum_zp = 0;
        float box_scale = 0, score_scale = 0, score_sum_scale = 0;

        for (int index = 0; index < headNum; index++)
        {
            int8_t *box_tensor = (int8_t *)pBlob[index * 3 + 0];
            int8_t *score_tensor = (int8_t *)pBlob[index * 3 + 1];
            int8_t *score_sum_tensor = (int8_t *)pBlob[index * 3 + 2];
            box_zp = qnt_zp[index * 3 + 0];
            score_zp = qnt_zp[index * 3 + 1];
            score_sum_zp = qnt_zp[index * 3 + 2];
            box_scale = qnt_scale[index * 3 + 0];
            score_scale = qnt_scale[index * 3 + 1];
            score_sum_scale = qnt_scale[index * 3 + 2];
            grid_h = mapSize[index][0];
            grid_w = mapSize[index][1];
            int grid_len = grid_h * grid_w;
            int8_t score_thres_i8 = qnt_f32_to_affine(objectThreshold, score_zp, score_scale);
            int8_t score_sum_thres_i8 = qnt_f32_to_affine(objectThreshold, score_sum_zp, score_sum_scale);

            for (int i = 0; i < grid_h; i++)
            {
                for (int j = 0; j < grid_w; j++)
                {
                    int offset = i * grid_w + j;
                    int max_class_id = -1;
                    if (score_sum_tensor[offset] < score_sum_thres_i8)
                        continue;
                    int8_t max_score = -score_zp;
                    for (int c = 0; c < model_class_num; c++)
                    {
                        if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score))
                        {
                            max_score = score_tensor[offset];
                            max_class_id = c;
                        }
                        offset += grid_len;
                    }

                    if (max_score > score_thres_i8)
                    {
                        offset = i * grid_w + j; // i = 0, j = 0 , offset = 0
                        float box[4];
                        for (int k = 0; k < 4; k++)
                        {
                            box[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                            offset += grid_len;
                        }
                        float x1, y1, x2, y2, w, h;
                        x1 = (-box[0] + j + 0.5) * strides[index];
                        y1 = (-box[1] + i + 0.5) * strides[index];
                        x2 = (box[2] + j + 0.5) * strides[index];
                        y2 = (box[3] + i + 0.5) * strides[index];                 
                        w = x2 - x1;
                        h = y2 - y1;                            
                        nn_object_s temp;
                        temp.x = x1;
                        temp.y = y1;
                        temp.w = w;
                        temp.h = h;
                        temp.score = deqnt_affine_to_f32(max_score, score_zp, score_scale);
                        temp.class_id = max_class_id;
                        detectRects.push_back(temp);         
                    }  
                }
            }
        }

        std::sort(detectRects.begin(), detectRects.end(),
                  [](nn_object_s &Rect1, nn_object_s &Rect2) -> bool
                  { return (Rect1.score > Rect2.score); });

        NN_LOG_DEBUG("NMS Before num :%ld", detectRects.size());
        for (int i = 0; i < detectRects.size(); ++i)
        {
            float xmin1 = detectRects[i].x;
            float ymin1 = detectRects[i].y;
            float xmax1 = detectRects[i].x + detectRects[i].w;
            float ymax1 = detectRects[i].y + detectRects[i].h;
            int classId = detectRects[i].class_id;
            float score = detectRects[i].score;
            if (classId != -1)
            {
                DetectiontRects.push_back(float(classId));
                DetectiontRects.push_back(float(score));
                DetectiontRects.push_back(float(xmin1));
                DetectiontRects.push_back(float(ymin1));
                DetectiontRects.push_back(float(xmax1));
                DetectiontRects.push_back(float(ymax1));
                for (int j = i + 1; j < detectRects.size(); ++j)
                {
                    float xmin2 = detectRects[j].x;
                    float ymin2 = detectRects[j].y ;
                    float xmax2 = detectRects[j].x + detectRects[j].w;
                    float ymax2 = detectRects[j].y + detectRects[j].h;
                    float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                    if (iou > nmsThreshold)
                    {
                        detectRects[j].class_id = -1;
                    }
                }
                
            }
        }  
        return ret;
    }

    // float
    int GetResult(float **pBlob, std::vector<float> &DetectiontRects)
    {
        int ret = 0;
        int grid_h, grid_w;
        int model_class_num = class_num;
        std::vector<nn_object_s> detectRects;
        for (int index = 0; index < headNum; index++)
        {
            float *box_tensor = (float *)pBlob[index * 3 + 0];         // 0, 3, 6 is box_tensor
            float *score_tensor = (float *)pBlob[index * 3 + 1];         // 1, 4, 7 is scores
            float *score_sum_tensor = (float *)pBlob[index * 3 + 2];       // 2, 5, 8 is scores sum
            grid_h = mapSize[index][0];
            grid_w = mapSize[index][1];
            int grid_len = grid_h * grid_w;
            for (int i = 0; i < grid_h; i++)
            {
                for (int j = 0; j < grid_w; j++)
                {
                    int offset = i * grid_w + j;   // i = 0, j = 0 , offset = 0
                    int max_class_id = -1;
                    if (score_sum_tensor[offset] < objectThreshold)
                        continue;

                    float max_score = 0;
                    for (int c = 0; c < model_class_num; c++)
                    {
                        if ((score_tensor[offset] > objectThreshold) && (score_tensor[offset] > max_score))
                        {
                            max_score = score_tensor[offset];
                            max_class_id = c;
                        }
                        offset += grid_len;

                    }             
                    if (max_score > objectThreshold)
                    {
                        offset = i * grid_w + j; // i = 0, j = 0 , offset = 0
                        float box[4];
                        for (int k = 0; k < 4; k++)
                        {
                            box[k] = box_tensor[offset];
                            offset += grid_len;
                        }
                        float x1, y1, x2, y2, w, h;
                        x1 = (-box[0] + j + 0.5) * strides[index];
                        y1 = (-box[1] + i + 0.5) * strides[index];
                        x2 = (box[2] + j + 0.5) * strides[index];
                        y2 = (box[3] + i + 0.5) * strides[index];                 
                        w = x2 - x1;
                        h = y2 - y1;                            
                        nn_object_s temp;
                        temp.x = x1;
                        temp.y = y1;
                        temp.w = w;
                        temp.h = h;
                        temp.score = max_score;
                        temp.class_id = max_class_id;
                        detectRects.push_back(temp);         
                    }         
                }
            }
        }

        std::sort(detectRects.begin(), detectRects.end(),
                  [](nn_object_s &Rect1, nn_object_s &Rect2) -> bool
                  { return (Rect1.score > Rect2.score); });

        NN_LOG_DEBUG("NMS Before num :%ld", detectRects.size());
        for (int i = 0; i < detectRects.size(); ++i)
        {
            float xmin1 = detectRects[i].x;
            float ymin1 = detectRects[i].y;
            float xmax1 = detectRects[i].x + detectRects[i].w;
            float ymax1 = detectRects[i].y + detectRects[i].h;
            int classId = detectRects[i].class_id;
            float score = detectRects[i].score;
            if (classId != -1)
            {
                DetectiontRects.push_back(float(classId));
                DetectiontRects.push_back(float(score));
                DetectiontRects.push_back(float(xmin1));
                DetectiontRects.push_back(float(ymin1));
                DetectiontRects.push_back(float(xmax1));
                DetectiontRects.push_back(float(ymax1));
                for (int j = i + 1; j < detectRects.size(); ++j)
                {
                    float xmin2 = detectRects[j].x;
                    float ymin2 = detectRects[j].y ;
                    float xmax2 = detectRects[j].x + detectRects[j].w;
                    float ymax2 = detectRects[j].y + detectRects[j].h;
                    float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                    if (iou > nmsThreshold)
                    {
                        detectRects[j].class_id = -1;
                    }
                }
                
            }
        }  
        return ret;
    }

}