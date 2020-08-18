#ifndef RRANSAC_H
#define RRANSAC_H


class RRANSACmodel;


class RRANSAC {
 public:
    cv::Mat A;      // State transition matrix (F)
    cv::Mat C;      // Measurement matrix (H)
    cv::Mat Q;      // Process Noise matrix
    cv::Mat R;      // Measurement Noise matrix

    cv::Mat O;      // Observability matrix
    cv::Mat G;      // propagated process noise matrix
    cv::Mat Qbig;   // Big matrix of block diagonal process noise
    cv::Mat Rbig;   // Big matrix of block diagoal measurement noise

    float dt;
    int Nw;         // Measurement window size
    int M;          // Number of stored models
    int ell;        // Number of RANSAC iterations
    float tauR;        // Inlier threshold
    float tau_rho;  // Good model threshold
    float gamma;    // Min number of inliers for a good model
    int U;          // Merge threshold
    int tau_T;      // Minimum number of time steps needed for a good model

    float pD_true;   // True probability of detection of all targets
    float lambdaGE;  // Average number of Gross errors per time Step

    std::vector<RRANSACmodel> models;
    std::vector<RRANSACmodel *> goodModels;
    std::vector<RRANSACmodel *> noGoodModels;

    std::vector<RRANSACmodel> tracks;  // available ended traces from goodModels;

    RRANSAC();
    void setup(int _M, int _U, float _tau_rho, int _tau_T,
        int _Nw, int _ell, float _tauR, float Q, float Rmx, float Rmy);
    void mul_allRows_vec(cv::Mat &mat, cv::Mat &vec);
    void apply(cv::Mat * T, cv::Mat * Z, cv::Mat * N, int t, int Ncentroids, int frameN);
    void RconstantVelocity(cv::Mat * T, cv::Mat * Z, int left, cv::Mat * N, int Ncentroids, int frameN);
    void constantVelocity(cv::Mat * T, cv::Mat * Z, cv::Mat * N, int y_index,
        cv::Mat * xhat_new, cv::Mat * P_new, cv::Mat * CS_new);
    cv::Mat getAllxhats();
    void mergeModels();
    // cv::Mat get_ModelsById(cv::Mat &ids, int att);
    void calculateResults();
    void plotModels(cv::Mat * frame);
    void plotGate(cv::Mat * frame, cv::Mat & error,
        cv::Point position, cv::Scalar colour);
    void plotTrace(cv::Mat * frame, cv::Mat * trace, cv::Scalar color);
    void print_models_xhat();
    void print_models_rho();
    void print_models_T();  
    int set_dataInput(cv::Mat * Z, cv::Mat * T, cv::Mat * N,
        std::vector<cv::Point> * measurements, int Nw, int fnum);
};

void horzcat(cv::Mat & A, cv::Mat & B);
void hcatPointInMat(cv::Mat & mat, cv::Point p);
bool cmpModelsbyRho(RRANSACmodel const & a, RRANSACmodel const & b);
bool cmpModelsbyT(RRANSACmodel const & a, RRANSACmodel const & b);

#include <memory>

class RRANSACmodel {
 public:
    cv::Mat xhat;
    cv::Mat P;
    int T;
    float rho;
    cv::Mat CS;
    int trackNum;
    int trackT;
    cv::Mat trace;
    bool good;
    RRANSAC * container;
    cv::Scalar color;
    std::vector<int> framesN;


    RRANSACmodel(RRANSAC * parent);
    RRANSACmodel(RRANSAC * parent, cv::Mat x, cv::Mat p, int t, float r, 
        cv::Mat cs, int tn, int tt);
    void reset();
    void keepnTrace(cv::Mat & position);
};

void Kronecker(cv::Mat &A, cv::Mat &B, cv::Mat &dst);
cv::Mat matPow(cv::Mat mat, int exponent);
float sortOneElem(cv::Mat * mat);
bool isAllZero(cv::Mat * mat);
void copyIdxsRows(cv::Mat * src, cv::Mat * dst, cv::Mat * idxs);
void logic_isSmaller(cv::Mat * src, float param, cv::Mat * dst);
cv::Mat logic_isBigger(cv::Mat &A, float param);
void printMat(cv::Mat &mat);
void print_ptr_models_rho(std::vector<RRANSACmodel *> & vec);

#endif

