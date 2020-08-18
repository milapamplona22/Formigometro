#ifndef BGS_PROCESSING
#define BGS_PROCESSING
#include "opencv2/opencv.hpp"

namespace bgs
{
    /* Functions */
    void preprocessing(cv::Mat &frame, cv::Mat &gray,
        const cv::Size resz=cv::Size(), const cv::Mat &mask=cv::Mat())
    {
            // 1. mask
            if (!mask.empty())
                bitwise_and(frame, mask, frame);
            // 2. downsample
            cv::Mat resized, * ptr;
            if (resz != cv::Size()){
                cv::resize(frame, resized, resz);
                ptr = &resized;
            }
            else
                ptr = &frame;
            // 3. conv gaussian gray
            cv::GaussianBlur(*ptr, *ptr, cv::Size(5,5), 1.5);
            cv::cvtColor(*ptr, gray, CV_BGR2GRAY);
    }


    void findFilterDrawContours(cv::Mat * fg,
        std::vector<std::vector<cv::Point> > *contours, int flag, float minArea)
    {
        std::vector<cv::Vec4i> hierarchy;
        contours->clear();
        cv::findContours(*fg, *contours, hierarchy, cv::RETR_EXTERNAL, flag);
        fg->setTo(cv::Scalar(0));

        // remove contours whose area < minArea
        contours->erase(std::remove_if(contours->begin(), contours->end(),
                [&minArea](std::vector<cv::Point> &c){
                    return cv::contourArea(c) < minArea;
                }), contours->end());

        // draw all contours
        drawContours(*fg, *contours, -1, cv::Scalar(255), CV_FILLED, 8);
    }


    std::vector<std::vector<cv::Point> >
    postprocessing(cv::Mat &fg, float minArea, bool open=true, bool close=true,
                      int medianb=int(5))
    {
        std::vector<std::vector<cv::Point> > contours;
        if (open || close)
        {
            cv::Mat m_element3 = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
            if (open)
                cv::morphologyEx(fg, fg, cv::MORPH_OPEN, m_element3);
            if (close)
                cv::morphologyEx(fg, fg, cv::MORPH_CLOSE, m_element3);
        }

        findFilterDrawContours(&fg, &contours, cv::CHAIN_APPROX_SIMPLE, minArea);
        
        if (medianb > 0)
            medianBlur(fg, fg, medianb);

        findFilterDrawContours(&fg, &contours, cv::CHAIN_APPROX_NONE, minArea);
        return contours;
    }


    std::vector<cv::Point>
    getCentroids(cv::Mat &fg, float minArea,
        bool open=true, bool close=true, int medianb=5)
    {
        // get all contours from post processing
        std::vector<std::vector<cv::Point> > contours =
            postprocessing(fg, minArea, open, close, medianb);
        // get each contour's centroid
        std::vector<cv::Point> centroids(contours.size());
        std::transform(contours.begin(), contours.end(), centroids.begin(),
            [](std::vector<cv::Point> &contour){
                cv::Moments mo = cv::moments(contour, false);
                return cv::Point(mo.m10 / mo.m00,
                                 mo.m01 / mo.m00);
            });
        return centroids;
    }

    std::vector<cv::Point>
    getCentroids(std::vector<std::vector<cv::Point> > &contours)
    {
        std::vector<cv::Point> centroids(contours.size());
        std::transform(contours.begin(), contours.end(), centroids.begin(),
            [](std::vector<cv::Point> &contour){
                cv::Moments mo = cv::moments(contour, false);
                return cv::Point(mo.m10 / mo.m00,
                                 mo.m01 / mo.m00);
            });
        return centroids;
    }

} // close namespace bgs

#endif //BGS_PROCESSING