#include "common.h"

using namespace std;
    using namespace cv;
    static const char *TAG = "[opencv_img2Gray]";

    napi_value Img2Gray(napi_env env, napi_callback_info info) {
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "Img2Gray Begin");
        napi_value result = NapiGetNull(env);
        size_t argc = 3;
        napi_value argv[3] = {nullptr};

        napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

        size_t strSize;
        char strBuf[256];
        napi_get_value_string_utf8(env, argv[1], strBuf, sizeof(strBuf), &strSize);
        std::string fileDir(strBuf, strSize);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "fileDir：%{public}s", fileDir.c_str());

        napi_get_value_string_utf8(env, argv[2], strBuf, sizeof(strBuf), &strSize);
        std::string fileName(strBuf, strSize);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "fileName：%{public}s", fileName.c_str());

        Mat srcImage;
        if (!GetMatFromRawFile(env, argv[0], fileDir, fileName, srcImage)) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "Get Mat from rawfile failed!.");
            return result;
        }

        Mat srcGray;
        cvtColor(srcImage, srcGray, COLOR_RGB2GRAY);

        // 將图像转换为pixelMap格式
        Mat outMat;
        cvtMat2Pixel(srcGray, outMat, RGBA_8888);
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "outMat size: %zu, cols：%{public}d, rows：%{public}d",
                     outMat.total(), outMat.cols, outMat.rows);

        napi_create_object(env, &result);
        bool retVal = WrapJsPixelInfoInfo(env, outMat, result);
        if (!retVal) {
            OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "WrapJsInfo failed!.");
        }

        return result;
    }
