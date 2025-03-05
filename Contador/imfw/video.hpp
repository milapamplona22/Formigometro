#ifndef VIDEO_HPP
#define VIDEO_HPP
#include "opencv2/opencv.hpp"
#include <string>

class Video : public cv::VideoCapture {
 public:
    Video(){ count_ = -1; }
    Video(int cam) : camera(true){ 
        open(cam);
        getMainInfos();
        count_ = -1;
    }
    Video(char * filename_) : filename(filename_){
        camera = false;
        open(filename);
        getMainInfos();
        count_ = -1;
    }
    Video(const std::string &filename_) : filename(filename_){
        camera = false;
        open(const_cast<char *>(filename.c_str()));
        getMainInfos();
        count_ = -1;
    }
    bool isCam(){ return camera; }
    cv::Size size(){ return size_; }
    double fps(){ return fps_; }
    std::string name(){ return filename; }
    unsigned int length(){ return length_; }
    unsigned int pos(){ return count_; }
    cv::VideoCapture& operator >> (cv::Mat& image);
 protected:
    void getMainInfos(){
        int height = (int) get(cv::CAP_PROP_FRAME_HEIGHT);
        int width = (int) get(cv::CAP_PROP_FRAME_WIDTH);
        size_ = cv::Size(width, height);
        fps_ = get(cv::CAP_PROP_FPS);
        length_ = get(cv::CAP_PROP_POS_FRAMES);
    }
    std::string filename;
    bool camera;
    cv::Size size_;
    double fps_;
    unsigned int length_;
    unsigned int count_;
};
cv::VideoCapture& Video::operator >> (cv::Mat& image){
        cv::VideoCapture::read(image);
        count_++;
        return *this;
}

#endif // VIDEO_HPP