#ifndef VIBE_HPP
#define VIBE_HPP
#include "opencv2/opencv.hpp"
#include "ViBeModel.hpp"

class ViBe : public VibeSDK::ViBeModel {
 public:
    bool apply(cv::Mat &gray, cv::Mat &fg){
        if (!initDone) {
            VibeSDK::RawImage refImg(gray.cols, gray.rows,
                gray.cols, 1, gray.data, false);
            init(&refImg);
            return false;
        }
        else {
            updateModel(gray.data);
            fg = cv::Mat(gray.rows, gray.cols, CV_8UC1,
                getLastComputedMask());
            return true;
        }
    }
};

#endif // VIBE_HPP