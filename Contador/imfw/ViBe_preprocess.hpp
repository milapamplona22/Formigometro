#ifndef VIBE_PREPROCESS_HPP
#define VIBE_PREPROCESS_HPP
#include "opencv2/opencv.hpp"
#include "ViBe.hpp"

void ViBe_preProcess(cv::VideoCapture &video, ViBe &vibe, int n,
    const cv::Mat mask=cv::Mat(), const cv::Size resize2=cv::Size(), 
    const cv::Rect2d &roi=cv::Rect2d())
{
    cv::Mat frame, gray, fg;
    double original_pos = video.get(CV_CAP_PROP_POS_FRAMES);
    int k = (int) original_pos;
    // for (int i = 0; i < n*2; i++){ 
    //     int j = i >= n ? k + (n-(i-n)-1) : k + i-1; // vai e vem
    for (int j = k + n; j > k; j--) {
        video.set(cv::CAP_PROP_POS_FRAMES, j);
        video.read(frame); // don't use '>>' because Video.hpp child's class
                           // counter iterates on >> usage
        if (!mask.empty())
            bitwise_and(frame, mask, frame);
        if (resize2 != cv::Size())
            resize(frame, frame, resize2);
        if (roi != cv::Rect2d())
            frame = frame(roi);
        if (frame.empty())
            break;
        cv::cvtColor(frame, gray, CV_BGR2GRAY);
        vibe.apply(gray, fg);
    }
    video.set(cv::CAP_PROP_POS_FRAMES, original_pos);
}

#endif // VIBE_PREPROCESS_HPP
