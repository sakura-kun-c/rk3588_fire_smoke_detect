#ifndef RK3588_LOGGING_H
#define RK3588_LOGGING_H

// a logging wrapper so it can be easily replaced
#include <stdio.h>

// log level from low to high
// 0: no log
// 1: error
// 2: error, warning
// 3: error, warning, info
// 4: error, warning, info, debug
static int32_t g_log_level = 4;

// a printf wrapper so the msg can be formatted with %d %s, etc.

#define NN_LOG_ERROR(...)          \
    do                             \
    {                              \
        if (g_log_level >= 1)      \
        {                          \
            printf("[RKNN_ERROR] "); \
            printf(__VA_ARGS__);   \
            printf("\n");          \
        }                          \
    } while (0)

#define NN_LOG_WARNING(...)          \
    do                               \
    {                                \
        if (g_log_level >= 2)        \
        {                            \
            printf("[RKNN_WARNING] "); \
            printf(__VA_ARGS__);     \
            printf("\n");            \
        }                            \
    } while (0)

#define NN_LOG_INFO(...)          \
    do                            \
    {                             \
        if (g_log_level >= 3)     \
        {                         \
            printf("[RKNN_INFO] "); \
            printf(__VA_ARGS__);  \
            printf("\n");         \
        }                         \
    } while (0)

#define NN_LOG_DEBUG(...)          \
    do                             \
    {                              \
        if (g_log_level >= 4)      \
        {                          \
            printf("[RKNN_DEBUG] "); \
            printf(__VA_ARGS__);   \
            printf("\n");          \
        }                          \
    } while (0)

#endif // RK3588_LOGGING_H
