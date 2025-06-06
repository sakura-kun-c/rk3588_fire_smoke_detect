
#ifndef RK3588_DEMO_POSTPROCESS_H
#define RK3588_DEMO_POSTPROCESS_H

#include <stdint.h>
#include <vector>
#include "datatype.h"

namespace forlinx
{
    int GetResult(float **pBlob, std::vector<float> &DetectiontRects);                                                               
    
    int GetResultInt8(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects);
}

#endif // RK3588_DEMO_POSTPROCESS_H
