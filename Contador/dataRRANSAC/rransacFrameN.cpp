#include "opencv2/opencv.hpp"
#include <memory>
#include "rransacFrameN.hpp"
#include "myrand.h"
#include <math.h>

#include <limits>


RRANSAC::RRANSAC() {
    int deltaT = 1;
    setMyRand();
    pD_true  = 0.8;     // True probability of detection of all targets
    lambdaGE = 5;       // Average number of Gross errors per time Step
    dt = deltaT;

    // model Dynamics
    A = cv::Mat::eye(4, 4, CV_32FC1);
    A.at<float>(0, 2) = 1*dt;
    A.at<float>(1, 3) = 1*dt;
    C = cv::Mat::eye(2, 4, CV_32FC1);
    Q = cv::Mat::eye(4, 4, CV_32FC1);
    Q.at<float>(0, 0) = pow(dt, 4)/4; Q.at<float>(0, 2) = pow(dt, 3)/2;
    Q.at<float>(1, 1) = pow(dt, 4)/4; Q.at<float>(1, 3) = pow(dt, 3)/2;
    Q.at<float>(2, 0) = pow(dt, 3)/2; Q.at<float>(2, 2) = pow(dt, 2);
    Q.at<float>(3, 1) = pow(dt, 3)/2; Q.at<float>(3, 3) = pow(dt, 2);
    Q = pow(0.5, 2) * Q;
    R = cv::Mat::eye(2, 2, CV_32FC1)*(pow(2.5, 1));
    
    // Parameters
    Nw = 45;               // Measurement window size
    M = 10;                // Number of stored models
    ell = 20;             // Number of RANSAC iterations
    float maxR = *std::max_element(R.begin<float>(), R.end<float>());
    tauR = sqrt(maxR)*3;  // Inlier threshold
    tau_rho = 0.6;        // Good model threshold
    gamma = tau_rho * Nw;  // Min number of inliers for a good model
    U = 50;                // Merge threshold
    tau_T  = 5;           // Minimum number of time steps needed
                          //   for a good model

    O = cv::Mat::zeros(Nw * C.rows, A.cols, CV_32FC1);
    cv::Mat tmp;
    for (int k = 0; k < Nw; k++) {
        tmp = C*(matPow(A, k));
        // copy tmp to O
        for (int i = 0; i < tmp.rows; i++)
            for (int j = 0; j < tmp.cols; j++)
                O.at<float>(k*tmp.rows+i, j) = tmp.at<float>(i, j);
    }
    
    G = cv::Mat::zeros(Nw * C.rows, Nw * A.rows, CV_32FC1);
    for (int n = Nw; n >= 2; n--) {
        tmp = cv::Mat::zeros(G.rows, C.cols, CV_32FC1);
        O.rowRange(0, O.rows-2*(n-1)).copyTo(tmp.rowRange(2*(n-1), tmp.rows));
        tmp.copyTo(G.colRange((n-1)*C.cols, n*C.cols));
    }
    
    // Big matrix of block diagonal process noise
    tmp = cv::Mat::eye(Nw, Nw, CV_32FC1);
    Kronecker(tmp, Q, Qbig);

    cv::Mat Rvec = R.diag().clone();
    cv::Mat Rbigvec = cv::Mat::ones(Rvec.rows, 2, CV_32FC1);
    mul_allRows_vec(Rbigvec, Rvec);

    // Big matrix of block diagoal measurement noise
    tmp = Rbigvec.reshape(1, Rbigvec.rows * Rbigvec.cols);
    Rbig = cv::Mat::eye(Rbigvec.rows * Rbigvec.cols,
        Rbigvec.rows * Rbigvec.cols, CV_32FC1);
    mul_allRows_vec(Rbig, tmp);

    // models.reserve(M);
    for (int i = 0; i < M; i++) {
        RRANSACmodel m(this);
        models.push_back(m);
    }
    // Z.create();
    // T.create();
}

void RRANSAC::setup(int _M, int _U, float _tau_rho, int _tau_T,
        int _Nw, int _ell, float _tauR, float Qm, float Rmx, float Rmy) {
    M = _M;
    U = _U;
    tau_T = _tau_T;
    Nw = _Nw;
    ell = _ell;
    tauR = _tauR;

    int deltaT = 1;
    setMyRand();
    pD_true  = 0.99;     // True probability of detection of all targets
    lambdaGE = 25;       // Average number of Gross errors per time Step
    dt = deltaT;

    // model Dynamics
    A = cv::Mat::eye(4, 4, CV_32FC1);
    A.at<float>(0, 2) = 1*dt;
    A.at<float>(1, 3) = 1*dt;
    C = cv::Mat::eye(2, 4, CV_32FC1);
    Q = cv::Mat::eye(4, 4, CV_32FC1);
    Q.at<float>(0, 0) = pow(dt, 4)/4; Q.at<float>(0, 2) = pow(dt, 3)/2;
    Q.at<float>(1, 1) = pow(dt, 4)/4; Q.at<float>(1, 3) = pow(dt, 3)/2;
    Q.at<float>(2, 0) = pow(dt, 3)/2; Q.at<float>(2, 2) = pow(dt, 2);
    Q.at<float>(3, 1) = pow(dt, 3)/2; Q.at<float>(3, 3) = pow(dt, 2);

    if (Qm)
        Q = Q * Qm;
    else
        Q = pow(0.5, 2) * Q;
    if (Rmx && Rmy) {
        R.row(0) = R.row(0) * Rmx;
        R.row(1) = R.row(1) * Rmy;
    }
    else
        R = cv::Mat::eye(2, 2, CV_32FC1)*(pow(2.5, 1));

    // Parameters
    float maxR = *std::max_element(R.begin<float>(), R.end<float>());
    tauR = sqrt(maxR)* _tauR;  // Inlier threshold
    tau_rho = _tau_rho;
    gamma = tau_rho * Nw;  // Min number of inliers for a good model

    O = cv::Mat::zeros(Nw * C.rows, A.cols, CV_32FC1);
    cv::Mat tmp;
    for (int k = 0; k < Nw; k++) {
        tmp = C*(matPow(A, k));
        // copy tmp to O
        for (int i = 0; i < tmp.rows; i++)
            for (int j = 0; j < tmp.cols; j++)
                O.at<float>(k*tmp.rows+i, j) = tmp.at<float>(i, j);
    }

    G = cv::Mat::zeros(Nw * C.rows, Nw * A.rows, CV_32FC1);
    for (int n = Nw; n >= 2; n--) {
        tmp = cv::Mat::zeros(G.rows, C.cols, CV_32FC1);
        O.rowRange(0, O.rows-2*(n-1)).copyTo(tmp.rowRange(2*(n-1), tmp.rows));
        tmp.copyTo(G.colRange((n-1)*C.cols, n*C.cols));
    }

    // Big matrix of block diagonal process noise
    tmp = cv::Mat::eye(Nw, Nw, CV_32FC1);
    Kronecker(tmp, Q, Qbig);

    cv::Mat Rvec = R.diag().clone();
    cv::Mat Rbigvec = cv::Mat::ones(Rvec.rows, 2, CV_32FC1);
    mul_allRows_vec(Rbigvec, Rvec);

    // Big matrix of block diagoal measurement noise
    tmp = Rbigvec.reshape(1, Rbigvec.rows * Rbigvec.cols);
    Rbig = cv::Mat::eye(Rbigvec.rows * Rbigvec.cols,
        Rbigvec.rows * Rbigvec.cols, CV_32FC1);
    mul_allRows_vec(Rbig, tmp);

    // models.reserve(M);
    for (int i = 0; i < M; i++) {
        RRANSACmodel m(this);
        models.push_back(m);
    }
    // Z.create();
    // T.create();
}


void RRANSAC::mul_allRows_vec(cv::Mat &mat, cv::Mat &vec) {
    //  multiply each element of a mat row
    //  for vec element of the same row
    CV_Assert(mat.rows == vec.rows &&
              vec.cols == 1);

    for (int i = 0; i < mat.rows; i++) {
        float * m = (float *) mat.ptr<float>(i);
        float * v = (float *) vec.ptr<float>(i);
        for (int j = 0; j < mat.cols; j++)
            m[j] = m[j] * v[0];
    }
}


void RRANSAC::apply(cv::Mat * T, cv::Mat * Z, cv::Mat * N, int leaving,
    int Ncentroids, int frameN)
{
    // 1) get data window ::
    //      - start - frameN of the first frame in this window
    //      Z = all measurements within the time window (2(x,y), #measurements)
    //      T = it's respectives timestamps(== frameN in which measurement was
    //          observed) (1, #measurements)
    //      N = #measurements observed in this window frame (1, Nw)
    if (N->cols > 2) {
        // 2) how many measurements are leaving the time window
        RconstantVelocity(T, Z, leaving, N, Ncentroids, frameN);
    }
    calculateResults();
}


void RRANSAC::calculateResults( ) {
    goodModels.clear();
    noGoodModels.clear();
    for (int i = 0; i < (int)models.size(); i++) {
        if (models[i].rho > tau_rho && models[i].T >= tau_T) {
            models[i].good = true;
            goodModels.push_back(&models[i]);
        }
        else
            noGoodModels.push_back(&models[i]);
    }
}


void RRANSAC::RconstantVelocity(cv::Mat * T, cv::Mat * Z,
    int left, cv::Mat * N, int Ncentroids, int frameN) {
    // RRANSAC constant velocity
    // 1) Initialize label number for valid tracks
    // 2) Calculate the number of measurements in the measurement window
    // 3) Step 1:  Model Management and Prediction 
    //      Remove inliers from each model's consensus set that have left the meausrement window
    //      Update the inlier ratio
    //      Update the model lifetime counter
    //      Predict model states and covariance using Kalman filter prediction equations

    // 4) Step 2: UPDATE STEP
    //      Look at each measurement to determine whether it is an inlier to an
    //      existing model or if a new model should be created

    // Step 1:  Model Management and Prediction 
    int numTracks = 0;
    for (int i = 0; i < (int)models.size(); i++)
        if (models[i].trackNum > -1)
            numTracks = numTracks +1;
    //Calculate the number of measurements in the measurement window
    cv::Scalar total = sum(*N);
    int totalNumMeas = total[0];
    
    // Step 1:  Model Management and Prediction 
    //   Remove inliers from each model's consensus set that have left the meausrement window
    //   Update the inlier ratio
    //   Update the model lifetime counter
    //   Predict model states and covariance using Kalman filter prediction equations
    for (int i = 0; i < (int)models.size(); i++) {
        if (models[i].T != -1) {
            cv::Mat viableInliers = models[i].CS - left;
            models[i].CS = logic_isBigger(viableInliers, (float)-1);
            int numInliers = models[i].CS.cols; 
            models[i].rho = (float) numInliers / Nw;
            models[i].T = models[i].T + 1;
            if (models[i].trackT > 0)
                models[i].trackT++;
            // Prediction Step for all good models
            models[i].xhat = A * models[i].xhat;
            models[i].P = A * models[i].P * A.t() + Q;
        }
    }
    // Step 2: UPDATE STEP
    // Look at each new measurement to determine whether it is an inlier to an
    // existing model or if a new model should be created
    for (int j = 0; j < Ncentroids; j++) {
    //for (int j = 0; j < N->at<float>(0, N->cols-1); j++) {
        // Caculate the error vector for current measurement
        cv::Mat currMeas = Z->col(Z->cols - (int)N->at<float>(0,N->cols-1) + j);
        //std::cout<< "currMeas =\n" << currMeas <<std::endl;
        cv::Mat errorVec = cv::Mat::ones(Z->rows, (int)models.size(), CV_32FC1);
        mul_allRows_vec(errorVec, currMeas);
        cv::Mat models_xhats = getAllxhats();
        errorVec = errorVec - C * models_xhats;
        //std::cout<< "errorVec =\n" << errorVec <<std::endl;
        cv::Mat cmp = abs(errorVec) < cv::Mat::ones(errorVec.rows, errorVec.cols, CV_32F)*tauR;
        int inlier = false;

        // Determine if measurement is an inlier to an existing (viable) model
        for (int i = 0; i < errorVec.cols; i++) {
            cv::Scalar s = sum(abs(cmp.col(i)));
            if (s[0] == 255 * errorVec.rows) {  // 255*2
                inlier = true;
                // If a measurement is an inlier to one or more
                // models, update those models
                //  Perform Kalman update for each model
                //  and update where necessary
                // x_prev = models[i].xhat;
                // P_prev = models[i].P;
                // Measurement Update
                cv::Mat denomInv;
                solve(R + (C * models[i].P * C.t()),
                    cv::Mat::eye(2, 2, CV_32F), denomInv, cv::DECOMP_SVD);
                cv::Mat L = models[i].P * C.t() * denomInv;
                models[i].P = (cv::Mat::eye(4,4,CV_32F) - L * C)*models[i].P;
                models[i].xhat = models[i].xhat + L * errorVec.col(i);
                if (!models[i].CS.empty())
                    models[i].CS = models[i].CS.t();
                models[i].CS.push_back((float)(totalNumMeas-j+1));
                models[i].CS = models[i].CS.t();
                int numInliers = models[i].CS.cols;
                models[i].rho = (float) numInliers/Nw;
                //cv::Mat pos = currMeas.rowRange(0,2);
                cv::Mat pos = models[i].xhat.rowRange(0,2);
                models[i].keepnTrace(pos);
                //Feats:: aqui o models[i] faz update usando 
                //  um feats referente ao mesmo blob (currMeas)
                models[i].framesN.push_back(frameN);
            }

        }
        if (!inlier) {
            if (totalNumMeas < 2) continue;
            cv::Mat xhat_new, P_new, CS_new;
            constantVelocity(T, Z, N, j, &xhat_new, &P_new, &CS_new);
            RRANSACmodel new_model(this);
            new_model.xhat     = xhat_new.clone();
            new_model.P        = P_new.clone();
            new_model.T        = 1;
            new_model.CS       = CS_new.clone();
            //new_model.CS       = cv::Mat::zeros(0, 0, CV_32F)
            new_model.rho      = (float)CS_new.cols/Nw;
            new_model.trackNum = -1;
            new_model.trackT   = 0;
            // Concatanate to existing models and prune later
            new_model.framesN.push_back(frameN);
            models.push_back(new_model);
        }
    }
    mergeModels();
    //now models is pruned
    
    // Step 4: Identify valid tracks and assign label to new tracks
    // Iterate through models and assign track numbers and kill
    // track as necessary
    for (int i = 0; i < M; i++) {
        if (models[i].rho > tau_rho &&
            models[i].trackNum < 0 && models[i].T >= tau_T) {
            numTracks = numTracks+1;
            models[i].trackNum = numTracks;
            models[i].trackT = models[i].T;
        }
    }
    /*printf ("\n\n\n models = \n");
    for (int i = 0; i < (int)models.size(); i++) {
       printf ("%d, %.4f, %d, %d __ (%.3f, %3.f)  ", models[i].T, models[i].rho,
           models[i].trackT, models[i].trackNum, models[i].xhat.at<float>(0, 0),
           models[i].xhat.at<float>(0, 1));
       printf (models[i].good ? "good\n" : "\n");
    }*/
}


void RRANSAC::constantVelocity(cv::Mat * T, cv::Mat * Z,
    cv::Mat * N, int y_index, cv::Mat * xhat_new, cv::Mat * P_new,
    cv::Mat * CS_new) {
    // Compute the number of scans with measurements
    int numScansWithMeas = 0;
    float * pN = (float *) N->ptr(0);
    for (int i = 0; i < N->cols; i++)
        if ( pN[i] > (float)0 )
            numScansWithMeas++;
    // Compute the number of measurements
    int numMeas = T->cols;
    // Convert the time stamp to an 1-indexed variable from start of window
    // windIndx = windIndx - (windIndx(end) - PP.Nw +1) +1
    cv::Mat windIndx = *T - (T->at<float>(0, T->cols-1) - Nw +1) +1;
    // Number of points needed for the minimum subset
    int p = 2;
    // Construct Observability matrix
    //  NiedBeard2014 --> \mathcal{O} matrix defined below Equation 3
    cv::Mat newO(Z->rows * Z->cols, C.cols, CV_32F);
    for (int j = numMeas-1; j >= 0; j--) {
        //  O(2*j-1:2*j,:) = PP.O(2*windIndx(j)-1:2*windIndx(j),:);
        //  -1 C is zero based;
        int val = 2*(int)windIndx.at<float>(0, j)-1;
        //  +1 & +2 because col|rowRange upper limit is exclusive
        O.rowRange(val-1, val+1).copyTo(newO.rowRange(2*j, 2*j+2));
    }
    // Reshape the Yvec to a pure vector
    cv::Mat Zvec = Z->t(); // opencv precedence is for cols
    Zvec = Zvec.reshape(0, Z->cols*Z->rows);
    // reshape in Matlab has row precedence
    // so it's always trouble when converting to opencv
    int maxNumInliers = 0;
    cv::Mat CS;
    cv::Mat Obar_best, Sbar_inv_best;
    cv::Mat x_prime_best(xhat_new->size(), CV_32F);
    for (int i = 0; i < ell; i++) {
        int indexSet = 0, val = 0;
        // Randomly select scans to choose measurements from
        if ( !isAllZero(N) ) {
            while ( val == 0 ) {
                // indexSet(1) = scansWithMeas(randi(numScansWithMeas-1));
                // => aqui tem que ser -2 e tem que iniciar em 0
                // verifiquei isso a exaustao
                indexSet = (int) myrand_uniform(0, N->cols-2);
                val = N->at<float>(0, indexSet);
            }
        } else printf("\n\nERROR: N is all zero\n\n");
        // Find Measurement Indices
        // abaixo nao precisa de -1 porque indexSet[0,end-2]
        cv::Scalar s = sum(N->colRange(0, indexSet));
        int minimumSubset1 = (int) s[0] + myrand_uniform(1, val);
        int minimumSubset2 = Z->cols - (int) N->at<float>(0, N->cols-1) + y_index;
        //   // Define indicator matrix
        //   //   NiedBeard2014 --> Equation 13
        //   M = zeros(p*2,2*numMeas);
        //   for j = p:-1:1
        //     M(2*j-1:2*j,2*minimumSubset(j)-1:2*minimumSubset(j)) = eye(2);
        //   endl

        // Construct Propagated Process Noise Matrix
        //   NiedBeard2014 --> G matrix defined below Equation 3
        //   Notice that we account for the multiple measurements per scan
        //  using windIndx. Much faster than doing several
        //  matrix multiplications
        cv::Mat Gbar = cv::Mat::zeros(C.cols, G.cols, CV_32F);
        int col_pos1 = 2*windIndx.at<float>(0, minimumSubset1-1);
        int col_pos2 = 2*windIndx.at<float>(0, minimumSubset2);
        G.rowRange(col_pos1-2, col_pos1).copyTo(Gbar.rowRange(0, 2));
        G.rowRange(col_pos2-2, col_pos2).copyTo(Gbar.rowRange(2, 4));
        cv::Mat Sbar = (Gbar * Qbig * Gbar.t()) + Rbig;
        // Find Initial States
        cv::Mat MeasIndices(1, 4, CV_32F);
        float * pMI = MeasIndices.ptr<float>(0);
        pMI[0] = (float) 2*minimumSubset1-2;//zero based
        pMI[1] = (float) 2*minimumSubset1-1;//zero based
        pMI[2] = (float) 2*minimumSubset2-2;//zero based
        pMI[3] = (float) 2*minimumSubset2-1;//zero based
        cv::Mat Obar(4, 4, CV_32F);
        for (int k = 0; k < MeasIndices.cols; k++)
              newO.row(pMI[k]).copyTo(Obar.row(k));
        cv::Mat Sbar_inv;
        solve(Sbar, cv::Mat::eye(p*2, p*2, CV_32F), Sbar_inv, cv::DECOMP_SVD);
        cv::Mat temp = Obar.t() * Sbar_inv;
        
        //  NiedBeard2014 --> Equation 16, where Yvec(MeasIndices) = M*Yvec
        cv::Mat Zvec_measIdx(4, Zvec.cols, CV_32F);
        for (int k = 0; k < MeasIndices.cols; k++ )
              Zvec.row(pMI[k]).copyTo(Zvec_measIdx.row(k));
        cv::Mat x_prime;
        solve((temp*Obar), (temp*Zvec_measIdx), x_prime, cv::DECOMP_SVD);
        cv::Mat errorVec = Zvec - newO*x_prime;
        cv::Mat error(2, numMeas, CV_32F);
        // istead of error = reshape(errorVec,2,numMeas)
        for (int k = 0; k < error.cols; k++)
           errorVec.rowRange(k*2, (k+1)*2).copyTo(error.col(k));
        // tested;; same reshape trouble between MATLAB and openCV
        sqrt(error.row(0).mul(error.row(0)) +
            error.row(1).mul(error.row(1)), error);
        error = abs(error);
        cv::Mat isInlier;
        logic_isSmaller(&error, (float)tauR, &isInlier);
        cv::Scalar sum_numInliers = sum(isInlier);
        int numInliers = sum_numInliers[0];
        
        // Keep track of the best model so far
        if (numInliers > maxNumInliers) {
            maxNumInliers = numInliers;
            CS.create(1, numInliers, CV_32F);
            pMI = (float *) isInlier.ptr<float>(0);
            pN = (float *) CS.ptr<float>(0);
            for (int k = 0, n = 0; k < isInlier.cols; k++) {
                if ( pMI[k] == (float) 1 )  {
                    pN[n] = (float) k;
                    n++;
                }
            }
            Obar.copyTo(Obar_best);
            Sbar_inv.copyTo(Sbar_inv_best);
            x_prime.copyTo(x_prime_best);
        }
        if (maxNumInliers > gamma)
            break;
    }
        
    // Propagate best hypothesis of initial states forward using Kalman filter
    // Update using only measurements in the consensus set at the appropriate times 
    cv::Mat x(x_prime_best.size(), CV_32F);
    x_prime_best.copyTo(x);
    // NiedBeard2014 --> Equation 17
    cv::Mat P;
    if (!Obar_best.empty()) 
        solve((Obar_best.t()*Sbar_inv_best*Obar_best),
            cv::Mat::eye(p*2, p*2, CV_32F), P);

    int startIndx = 1;
    int endIndx = 0;
    for (int i = 0; i < std::min(numScansWithMeas, Nw); i++) {
        // We don't need to predict on the first step
        if (i > 0) {
            x = A * x;
            P = (A * P * A.t()) + Q;
        }
        // Determine if there are inlier measurements 
        // in this scan and update with them
        endIndx = endIndx + (int)N->at<float>(0, i);
        cv::Mat inliers = cv::Mat::zeros(1, CS.cols, CV_32F);
        bool any_inlier = false;
        pN = (float *) CS.ptr<float>(0);
        float * pIn = (float *) inliers.ptr<float>(0);
        for (int k = 0; k < CS.cols; k++) {
            // +1 in each CS value to be compatible with matlab
            if ( pN[k]+1 >= (float) startIndx && pN[k]+1 <= (float) endIndx ) {
                pIn[k] = (float) 1;
                any_inlier = true;
            }
        }
        if ( any_inlier ) {
            // for j
            float * pIn = (float *) inliers.ptr<float>(0);
            for (int k = 0; k < inliers.cols; k++) {
                if ( pIn[k] == (float) 1 ) {
                    cv::Mat denomInv;
                    solve((R +(C * P* C.t())), cv::Mat::eye(2, 2, CV_32F), denomInv);
                    cv::Mat L = P * C.t() * denomInv;
                    P = (cv::Mat::eye(4, 4, CV_32F) - (L * C)) * P;
                    int j = (int)CS.at<float>(0, k);
                    x = x + L * (Z->col(j) - (C * x));
                }
            }
        }
        startIndx = startIndx + (int)N->at<float>(0, i);
    }
    x.copyTo(* xhat_new);
    P.copyTo(* P_new);
    CS.copyTo(* CS_new);
}


void RRANSAC::mergeModels() {
    models.erase(std::remove_if(models.begin(), models.end(), [](const RRANSACmodel& model)
        {return model.T == -1;}), models.end());
    std::sort(models.begin(), models.end(), cmpModelsbyRho);
    
    for (int m = 0; m < (int)models.size(); m++) {
        std::vector<int> similar;
        for (int k = 0; k < (int)models.size(); k++) {
            if (k != m) {
                cv::Mat invP = models[k].P.inv();
                cv::Mat diff = models[k].xhat - models[m].xhat;
                cv::Mat dist(1, 1, CV_32F);  // Mahalanobis distance^2
                dist = diff.t() * invP * diff;
                if ( dist.at<float>(0, 0) < U )
                    similar.push_back(k);
            }
        }
        std::vector<RRANSACmodel *> mask;
        int sum_trackNum = 0;
        for (int i = 0; i < (int)similar.size(); i++) {
            if (models[similar[i]].trackNum > 0) {
                mask.push_back(&models[similar[i]]);
                sum_trackNum = sum_trackNum + models[similar[i]].trackNum;
            }
        }
        if (sum_trackNum > 1) {
            std::vector<int> bestLifetime;
            int max_v = -1;
            for (int i = 0; i < (int)mask.size(); i++) {
                if ( mask[i]->trackT > max_v )
                    max_v = mask[i]->trackT;
            }
            for (int i = 0; i < (int)mask.size(); i++) {
                if ( mask[i]->trackT == max_v )
                    bestLifetime.push_back(i);
            }

            int max_rho_i = 0;
            for (int i = 0; i < (int)bestLifetime.size(); i++) {
                float max_rho = -1;
                // get max_rho within mask(bestLifetime)
                for (int ii = 0; ii < (int)bestLifetime.size(); ii++)
                    if ( mask[bestLifetime[ii]]->rho > max_rho )
                        max_rho = mask[bestLifetime[ii]]->rho;

                // max_rho_i = first i whose bestLifetime[i] == max(bestLifetime.rho)
                for (int ii = 0; ii < (int)bestLifetime.size(); ii++) {
                    if ( mask[bestLifetime[ii]]->rho == max_rho ) {
                        max_rho_i = ii;
                        break;
                    }
                }
            }
            models[m].trackNum = mask[bestLifetime[max_rho_i]]->trackNum;
        }
        else {
            if ( sum_trackNum > 0 ) {
                // necessariamente size=1 porque mask[0].tracknNum = 1;
                models[m].trackNum = mask[0]->trackNum;
            }
            // else
                // No Track number assigned yet
        }   
        for (int e = 0; e < (int)similar.size(); e++) {
            if (models[similar[e]].good)
                tracks.push_back(models[similar[e]]);
            models.erase(models.begin()+similar[e]);
            for (int ee = 0; ee < (int)similar.size(); ee++) {
                if (similar[ee] > 0)
                    similar[ee] = similar[ee]-1;
            }
        }
    }
    // Fill rest of the bank of models with empty models as necessary
    if ( (int)models.size() < M ) {
        for (int i = (int)models.size(); i < M; i++) {
            if ((int)models.size() > i)
                models[i] = RRANSACmodel(this);
            else
                models.push_back(RRANSACmodel(this));
        }
    }
    // Prune extra Models (sort by rho)
    //   I will have to update this, because currently, I loose the track
    //   management in some cases -- ie, if model 2 dies, then model 3 shifts to
    //   take its place
    //   However, this code will work
    std::vector<RRANSACmodel> tempModelsBorder;
    std::sort(models.begin(), models.end(), cmpModelsbyRho);
    float thresh_rho = models[M-1].rho;
    for (int k = 0; k < (int)models.size(); k++) {
        if (models[k].rho == thresh_rho) {
            tempModelsBorder.push_back(models[k]);
            models.erase(models.begin()+k);
            k = k-1;
        }
    }
    std::sort(tempModelsBorder.begin(), tempModelsBorder.end(), cmpModelsbyT);
    std::reverse(tempModelsBorder.begin(), tempModelsBorder.end());
    for (int i = (int)models.size(); i <  M; i++) {
        models.push_back(tempModelsBorder.back());
        tempModelsBorder.pop_back();
    }
    tempModelsBorder.clear();
    // now RRANSAC.models is ok!
}


void RRANSAC::plotModels(cv::Mat * frame) {
    cv::Point center;
    cv::Mat Umat = cv::Mat::eye(2, 2, CV_32F)*U;
    for (int i = 0; i < (int)models.size(); i++) {
        center.x = (int) models[i].xhat.at<float>(0, 0);
        center.y = (int) models[i].xhat.at<float>(1, 0);
        if (models[i].good == true) {
            circle(* frame, center, 10, models[i].color, 3, 8);
            putText(* frame, std::to_string(models[i].trackNum), center, cv::FONT_HERSHEY_SIMPLEX, 1,
                models[i].color, 1, 4 );
            //plotGate(frame, R, center, cv::Scalar(0,0,0)); // R 
            cv::Mat _p = models[i].P.rowRange(0,2).colRange(0,2);
            plotGate(frame, _p, center, cv::Scalar(255,255,255)); // P (estimate covariance)
            plotGate(frame, Umat, center, cv::Scalar(250,50,50)); // U
            plotTrace(frame, &models[i].trace, models[i].color);
        }
        else {
            circle(* frame, center, 10, cv::Scalar(50,50,50), 3, 8);    
        }
    }
}

void RRANSAC::plotGate(cv::Mat * frame, cv::Mat &error,
    cv::Point position, cv::Scalar colour) {
    // see algorthm_research/plotMultiGaussian_poc
    float chisquare_val = 5.9915;
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(error, eigenvalues, eigenvectors);
    double angle = atan2(-1*eigenvectors.at<float>(0,1), -1*eigenvectors.at<float>(0,0));
    if (angle < 0)
        angle += 2*CV_PI;
    angle = 180.*angle/CV_PI;
    double majoraxis=2*sqrt(chisquare_val*eigenvalues.at<float>(0));
    double minoraxis=2*sqrt(chisquare_val*eigenvalues.at<float>(1));
    cv::RotatedRect el(position, cv::Size2f(majoraxis, minoraxis), angle);
    cv::ellipse(* frame, el, colour, 1);
}

void RRANSAC::plotTrace(cv::Mat * frame, cv::Mat * trace, cv::Scalar color) {
    int tailLength = 20;
    int tail = tailLength <= trace->rows ? 
            tailLength : trace->rows;
    
    cv::Point p1;
    cv::Point p2;
    for (int i = trace->rows-tail; i < trace->rows-1; i++) {
        p1.x = trace->at<float>(i,0);
        p1.y = trace->at<float>(i,1);
        p2.x = trace->at<float>(i+1,0);
        p2.y = trace->at<float>(i+1,1);
        line (* frame, p1, p2, color, 1, 8, 0);
    }
}


void horzcat(cv::Mat & A, cv::Mat & B) {
    if (B.cols > 0) {
        if (A.cols == 0)
            B.copyTo(A);
        else 
            cv::hconcat(A, B, A);
    }
}


void hcatPointInMat(cv::Mat & mat, cv::Point p) {
    float center[2];
    center[0] = p.x;
    center[1] = p.y;
    cv::Mat pm(2, 1, CV_32F, center);
    horzcat(mat, pm);
}


int RRANSAC::set_dataInput(cv::Mat * Z, cv::Mat * T, cv::Mat * N,
    std::vector<cv::Point> * measurements, int Nw, int fnum) {
        int numMeasLost = 0;
        int n_obs = (int)measurements->size();
        if ( n_obs > 0 ) {
            for (int i = 0; i < n_obs; i++)
                hcatPointInMat(*Z, (*measurements)[i]);

            cv::Mat t = cv::Mat::ones(1, n_obs, CV_32F)*fnum;
            cv::Mat n = cv::Mat::ones(1, 1, CV_32F)*n_obs;
            horzcat(*T, t);
            horzcat(*N, n);

            measurements->clear();
        }
        // Cortar por tempo
        if (!T->empty()) {
            float * pT = (float* )T->ptr<float>(0);
            float * pN = (float* )N->ptr<float>(0);
            if ( (float)fnum-pT[0] > Nw-1 ) {
                int meas_cut = 0;
                while ( (float)fnum-pT[meas_cut] > Nw-1 && meas_cut < T->cols)
                      meas_cut = meas_cut + 1;

                int cutN = 0;
                int soma = 0;
                while (soma < meas_cut) {
                    soma = soma+pN[cutN];
                    cutN = cutN+1;
                }
                * Z = Z->colRange(meas_cut, Z->cols);
                * T = T->colRange(meas_cut, T->cols);
                * N = N->colRange(cutN, N->cols);
                numMeasLost = meas_cut;
                // T.begin() changed in T = colRange, so reset pointer
                pT = (float* )T->ptr<float>(0);
            }
        }
        // Resetar os modelos se n~ao tive objs
        if (N->cols == 0) {
            for (int i = 0; i < (int)models.size(); i++) {
                if (models[i].good)
                    tracks.push_back(models[i]);
                if ( models[i].T != -1)
                    models[i] = RRANSACmodel(this);
            }
        }
        if (T->cols > 1) {
            float * lastT = T->ptr<float>(0);
            if ( (fnum > int(lastT[T->cols-1]+2)) && T->cols > 2 ){
                for (int i = 0; i < (int)models.size(); i++) {
                    if (models[i].good)
                        tracks.push_back(models[i]);
                    if (models[i].T != -1)
                        models[i] = RRANSACmodel(this);
                }
            }
        }
    return numMeasLost;
}


bool cmpModelsbyRho(RRANSACmodel const & a, RRANSACmodel const & b) {
    return ( a.rho > b.rho );
}


bool cmpModelsbyT(RRANSACmodel const & a, RRANSACmodel const & b) {
    return ( a.T > b.T );
}


cv::Mat RRANSAC::getAllxhats() {
    cv::Mat xhats(models[0].xhat.rows, (int)models.size(), CV_32F);
    for (int i = 0; i < (int)models.size(); i++)
        models[i].xhat.copyTo(xhats.col(i));

    return xhats;
}


RRANSACmodel::RRANSACmodel(RRANSAC * parent) {
    container = parent;
    xhat = cv::Mat(4, 1, CV_32FC1);
    for (int i = 0; i < 4; i++)
        xhat.at<float>(i, 0) = NAN;
    P    = cv::Mat::zeros(4, 4, CV_32FC1);
    T    = -1;
    rho  = 0;
    trackNum = -1;
    trackT   = 0;
    trace.create(0, container->C.rows, CV_32F);
    good = false;
    for (int i = 0; i < 3; i++)
        color[i] = (int) (255 * ((double)rand()/(double)RAND_MAX));
}


RRANSACmodel::RRANSACmodel(RRANSAC * parent, cv::Mat x, cv::Mat p, int t, float r,
    cv::Mat cs, int tn, int tt) {
    container = parent;
    xhat = x.clone();
    P = p.clone();
    T = t;
    rho = r;
    CS = cs.clone();
    trackNum = tn;
    trackT = tt;
    trace.create(0, container->C.rows, CV_32F);
    good = false;
    for (int i = 0; i < 3; i++)
        color[i] = (int) (255 * ((double)rand()/(double)RAND_MAX));
}


void RRANSACmodel::reset() {
    for (int i = 0; i < 4; i++)
        xhat.at<float>(i, 0) = NAN;
    P    = cv::Mat::zeros(4, 4, CV_32FC1);
    T    = -1;
    rho  = 0;
    trackNum = -1;
    trackT   = 0;
    trace.create(0, container->C.rows, CV_32F);
    good = false;
    for (int i = 0; i < 3; i++)
        color[i] = (int) (255 * ((double)rand()/(double)RAND_MAX));
}


void RRANSACmodel::keepnTrace(cv::Mat & position) {
    trace.resize(trace.rows+1, position.rows);
    cv::Mat row = position.col(0).t();
    row.copyTo(trace.row(trace.rows-1));
}


void Kronecker(cv::Mat &A, cv::Mat &B, cv::Mat &dst) {
    dst.create(A.rows * B.rows, A.cols * B.cols, CV_32FC1);
    int count = 0;

    for (int ia = 0; ia < A.rows; ia++) {
        float * pA = (float *) A.ptr<float>(ia);

        for (int ib = 0; ib < B.rows; ib++) {
            float * pB = (float *) B.ptr<float>(ib);

            for (int ja = 0; ja < A.cols; ja++) {
                float * pdst = (float *) dst.ptr<float>(count/(A.cols*B.cols));
                for (int jb = 0; jb < B.cols; jb++) {
                    pdst[ja * B.cols + jb] = pA[ja] * pB[jb];
                    count++;
                }
            }
        }
    }
}


cv::Mat matPow(cv::Mat mat, int exponent) {
    cv::Mat res = mat.clone();
    if ( exponent == 0 )
        return cv::Mat::eye(mat.rows, mat.cols, mat.type());

    for (int i = 1; i < exponent; i++)
        res = res * mat;

    return res;
}


float sortOneElem(cv::Mat * mat) {
    int idx = myrand_uniform(0, mat->cols-1);
    return mat->at<float>(0, idx);
}


bool isAllZero(cv::Mat * mat) {
    for (int i = 0; i < mat->rows; i++) {
        float * p = (float *) mat->ptr(i);
        for (int j = 0; j < mat->cols; j++) {
            if (p != 0)
                return false;
        }
    }
    return true;
}


void copyIdxsRows(cv::Mat * src, cv::Mat * dst, cv::Mat * idxs) {
    for (int i = 0; i < idxs->rows; i++) {
        float * p_idxs = (float *) idxs->ptr<float>(i);
        for (int j = 0; j < idxs->cols; j++) {
            src->row(p_idxs[j]).copyTo(dst->row(i));
        }
    }
}


void logic_isSmaller(cv::Mat * src, float param, cv::Mat * dst) {
    dst->create( src->size(), src->type());
    for (int i = 0; i < src->rows; i++) {
        float * s_ptr = (float *) src->ptr<float>(i);
        float * d_ptr = (float *) dst->ptr<float>(i);
        for (int j = 0; j < src->cols; j++) {
            if ( s_ptr[j] < param )
                d_ptr[j] = 1;
            else
                d_ptr[j] = 0;
        }
    }
}


cv::Mat logic_isBigger(cv::Mat &A, float param) {
    cv::Mat dst;
    for (int i = 0; i < A.rows; i++) {
        float * p = (float *) A.ptr<float>(i);
        for (int j = 0; j < A.cols; j++)
            if (p[j] > param)
                dst.push_back(p[j]);
    }
    if (!dst.empty())
        dst = dst.t();
    return dst;
}


void printMat(cv::Mat &mat) {
    printf("[");
    for (int i = 0; i < mat.rows; i++) {
        if (i > 0)
            printf(" ");
        float * p = (float *)mat.ptr<float>(i);
        for (int j = 0; j < mat.cols; j++) {
            printf("%5f ", p[j]);
        }
        if (i < mat.rows-1)
        printf("\n");
    }
    printf("]\n");
}

void RRANSAC::print_models_xhat() {
    cv::Mat xhats ( models[0].xhat.rows, (int)models.size(), CV_32F);
    for ( int i = 0; i < (int)models.size(); i++) 
        models[i].xhat.col(0).copyTo(xhats.col(i));
    std::cout<< "models[:].xhat =\n" << xhats <<std::endl;
}


void RRANSAC::print_models_rho() {
    cv::Mat rhos ( 1, (int)models.size(), CV_32F);
    float * p = (float *) rhos.ptr<float>(0);
    for ( int i = 0; i < (int)models.size(); i++) 
        p[i] = models[i].rho;
    std::cout<< "models[:].rho =\n" << rhos <<std::endl;
}


void RRANSAC::print_models_T() {
    cv::Mat Ts(1, (int)models.size(), CV_32F);
    float * p = (float *) Ts.ptr<float>(0);
    for ( int i = 0; i < (int)models.size(); i++) 
        p[i] = models[i].T;
    std::cout<< "models[:].T =\n" << Ts <<std::endl;
}


void print_ptr_models_rho(std::vector<RRANSACmodel *> & vec) {
    cv::Mat data(1, (int)vec.size(), CV_32F);
    float * p = (float *)data.ptr(0);
    for ( int i = 0; i < (int)vec.size(); i++)
        p[i] = vec[i]->rho;
    
    std::cout<< "vec.rho =\n" << data <<std::endl;
}


/*cv::Mat RRANSAC::get_ModelsById(cv::Mat &ids, int att) {
    int size = ids.rows * ids.cols;
    cv::Mat dst(size, 1, CV_32F);

    float * p = (float *) dst.data;
    int count = 0;
    for (int i = 0; i < ids.rows; i++) {
        float * id = (float *)ids.ptr<float>(i);
        for (int j = 0; j < ids.cols; j++) {
            switch (att) {
                case 'T':
                    p[count] = models[id[j]].T;
                    break;
                case 'rho':
                    p[count] = models[id[j]].rho;
                    break;
                case 'trackNum':
                    p[count] = models[id[j]].trackNum;
                    break;
                case 'trackT':
                    p[count] = models[id[j]].trackT;
                    break;
            }
            count++;
        }
    }
    return dst;
}*/
