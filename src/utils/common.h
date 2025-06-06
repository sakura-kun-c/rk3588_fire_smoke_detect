#ifndef COMMON_H_
#define COMMON_H_

#ifdef MAIN_C_
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <thread>
#include <map>
#include <vector>
#include "im2d.h"
#include "rga.h"
#include "RgaUtils.h"
#include "rknn_api.h"
#include "zlmediakit/zl_pull.h"
#include "zlmediakit/zl_push.h"
#include "utils/engine_helper.h"
#include "draw/cv_draw.h"
#include "threadpool/yolov8_flow.h"
#include "threadpool/yolov8_thread_pool.h"

#endif // MAIN_C_

#ifdef ZL_PUSH_C_
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>

#include "im2d.h"
#include "rga.h"
#include "RgaUtils.h"
#include "rknn_api.h"
#include "mk_mediakit.h"
#include "utils/datatype.h"

#endif // ZL_PUSH_C_

#endif // COMMON_H_