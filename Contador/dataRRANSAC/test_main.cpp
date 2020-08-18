#include <iostream>
#include "opencv2/opencv.hpp"
#include <array>
#include "rransacFeats.hpp"
// algorithm_research/code
#include "argParse.hpp"
// algorithm_research/imfw
#include "video.hpp"


using namespace cv;
using namespace std;


class Feats : public BaseFeats
{
 public:
    int n;
    vector<int> framesN;    
    Feats(int i) n(i) {}
    void update(const shared_ptr<BaseFeats> old, const shared_ptr<BaseFeats> newval)
    {
        shared_ptr<Feats> f_old = std::dynamic_pointer_cast<Feats>(old);
        shared_ptr<Feats> f_newval = std::dynamic_pointer_cast<Feats>(newval);
        f_newval->framesN.push_back(n);
        if (f_old) {
            printf (" %d ", (int) f_old->framesN.size());
            framesN.insert(framesN.end(), f_old->framesN.begin(), f_old->framesN.end());
        }
    }
};

struct Track
{
    vector<array<float, 2> > position;
    vector<int> frames;
};

struct Options
{
    string infs;
    string outfs;
    string displayVid; bool has_video;
    // RRANSAC params
    int M, Nw, ell, tau_T, tauR;
    float U, tau_rho, Qm, Rmx, Rmy;
};

// ---------------------------- Options
void menu(char ** argv) {
    printf ("  %s -data <data yml> -out <tracks yml> [RRANSAC params]\n\n"
            "  -data  \t name of yml file containing centroids and frames\n"
            "  -out   \t output yml file, will have tracks positions and frames\n"
            "  -display\t (optional) video file to plot over\n"
            "\n  RRANSAC params: (must have)\n"
            "  -M      \t (int) Number of stored models\n"
            "  -U,     \t (float) Merge threshold\n"
            "  -tau_rho\t (float) Good model quality threshold\n"
            "  -tau_T, \t (int) Minimum number of time steps needed for a good model\n"
            "  -Nw,    \t (int) Measurement window size\n"
            "  -ell,   \t (int) Number of RANSAC iterations\n"
            "  -tauR,  \t (int) Inlier threshold (multiplier)\n"
            "  -Qm,    \t (float) Q multiplier\n"
            "  -Rmx,   \t (float) Rx multiplier\n"
            "  -Rmy    \t (float) Ry multiplier\n\n", argv[0]);
}
bool parseOptions(Options &opt, int argc, char ** argv);
bool rransacOptions(RRANSAC &rransac, int argc, char **argv);


// ----------------- read from input yml
inline Size readSize(const cv::FileNode &fn);
template <typename T>
    vector<T> readVec(const cv::FileNode &fn);


// ----------- setting data for RRANSAC
//             input and getting output
void getCentroids(const FileNode &fn, vector<Point> &centroids, int colx=0, int coly=1);
void rransacUpdate(RRANSAC &rransac, Mat &Z, Mat &T, Mat &N,
    vector<Point> &centroids, vector<Feats> &feats, int frameCount);
vector<Track> getRRANSACendedTracks(RRANSAC &rransac);


// ---------------- Write to output yml
void writeParameters(FileStorage &fs, RRANSAC &rransac, Options &opt);
void writeTrack2yml(Track & track, FileStorage &fs);


// --------- display results over video
void display(cv::Mat &frame, RRANSAC &rransac, std::vector<Point> &centroids);



int main(int argc, char ** argv) {
    // Parse arguments
    Options opt;
    if (!parseOptions(opt, argc, argv))
        return 0;
    
    // Parse and set RRANSAC
    Mat Z, T, N;
    RRANSAC rransac;
    if (!rransacOptions(rransac, argc, argv))
        return 0;
    
    // Open input file
    FileStorage fs(opt.infs, FileStorage::READ);
    Size size = readSize(fs["size"]); // Read size (width, height)
    FileNode features = fs["features"]; // Read features [frame# features[]]
    FileNodeIterator it = features.begin(), it_end = features.end(); // set it


    // Open output file
    FileStorage outfs(opt.outfs, FileStorage::WRITE);
    outfs << "input" << opt.infs;
    outfs << "size" << size;
    writeParameters(outfs, rransac, opt); // write rransac parameters used
    outfs << "tracks" << "["; // open tracks section to write to


    // if display over video
    Video video;
    Mat frame(size, CV_8UC3);
    if (opt.has_video)
        video = Video(opt.displayVid);


    int totalTracks = 0;
    int frameN = -1;
    for (int i = 0; it != it_end; ++it, i++)
    {
        frameN = (int)(*it)["frame"]; // read frame n
        // read centroids 
        vector<Point> centroids;
        getCentroids((*it)["features"], centroids, 0, 1);
        
        // create feats (frameN) for each centroid seen
        vector<Feats> feats(centroids.size());
        for (size_t f = 0; f < feats.size(); f++)
            feats[f].framesN.push_back(frameN);
        // update RRANSAC
        rransacUpdate(rransac, Z, T, N, centroids, feats, frameN);
    
        // get finished tracks
        vector<Track> tracks = getRRANSACendedTracks(rransac);
        printf ("\n\n");
        //export tracks to outfile
        for (size_t t = 0; t < tracks.size(); t++) {
            writeTrack2yml(tracks[t], outfs);
            totalTracks++;
        }
        
        // if -display
        if (opt.has_video)
        {
            video.set(cv::CAP_PROP_POS_FRAMES, frameN);
            video >> frame;
            display(frame, rransac, centroids);
            imshow(opt.displayVid, frame);
            if (waitKey(0) == 27)
                break;
        }
    }
    
    // close in and out FileStorages
    fs.release();
    outfs << "]"; // close tracks[:
    outfs.release();

    std::cout<< "last processed frame: " << frameN <<std::endl;
    std::cout<< "total n of tracks: " << totalTracks <<std::endl;
    std::cout<< std::endl;
    return 0;
}



// ---------------------------- Options
bool parseOptions(Options &opt, int argc, char ** argv)
{
    bool complete = true;
    if (!parseArg(argc, argv, "-data", opt.infs)){ printf ("option -data missing\n"); complete = false; }
    if (!parseArg(argc, argv, "-out", opt.outfs)){ printf ("option -out missing\n"); complete = false; }
    opt.has_video = parseArg(argc, argv, "-display", opt.displayVid);
    return complete;
}

bool rransacOptions(RRANSAC &rransac, int argc, char **argv)
{
    int M, Nw, ell, tau_T;
    float U, tau_rho, Qm, Rmx, Rmy, tauR;
    bool complete = true;
    if (!parseArg(argc, argv, "-M", M)){ printf ("option -M missing\n"); complete = false; }
    if (!parseArg(argc, argv, "-U", U)){ printf ("option -U missing\n"); complete = false; }
    if (!parseArg(argc, argv, "-tau_rho", tau_rho)){ printf ("option -tau_rho missing\n"); complete = false; }
    if (!parseArg(argc, argv, "-tau_T", tau_T)){ printf ("option -tau_T missing\n"); complete = false; }
    if (!parseArg(argc, argv, "-Nw", Nw)){ printf ("option -Nw missing\n"); complete = false; }
    if (!parseArg(argc, argv, "-ell", ell)){ printf ("option -ell missing\n"); complete = false; }
    if (!parseArg(argc, argv, "-tauR", tauR)){ printf ("option -tauR missing\n"); complete = false; }
    if (!parseArg(argc, argv, "-Qm", Qm)){ printf ("option -Qm missing\n"); complete = false; }
    if (!parseArg(argc, argv, "-Rmx", Rmx)){ printf ("option -Rmx missing\n"); complete = false; }
    if (!parseArg(argc, argv, "-Rmy", Rmy)){ printf ("option -Rmy missing\n"); complete = false; }
    if (!complete)
        return false;
    rransac.setup(
        M,        // M: Number of stored models
        U,        // U: Merge threshold
        tau_rho,  // tau_rho: Good model threshold
        tau_T,    // tau_T: Minimum number of time steps needed for a good model
        Nw,       // Nw: Measurement window size
        ell,      // ell: Number of RANSAC iterations
        tauR,     // tauR: Inlier threshold (multiplier)
        Qm,       // Qm
        Rmx,      // Rmx
        Rmy       // Rmy
    );
    return true;
}

// ----------------- read from input yml
template <typename T>
vector<T> readVec(const cv::FileNode &fn)
{
    vector<T> vec;
    cv::FileNodeIterator it = fn.begin(), it_end = fn.end();
    for (; it != it_end; ++it)
        vec.push_back((T)*it);
    return vec;
}

inline Size readSize(const cv::FileNode &fn)
{
    vector<int> vsize = readVec<int>(fn);
    return Size(vsize[0], vsize[1]);
}


// ----------- setting data for RRANSAC
//             input and getting output
void getCentroids(const FileNode &fn, vector<Point> &centroids, int colx, int coly)
{
    FileNodeIterator it = fn.begin(), it_end = fn.end();
    int i = 0;
    for (; it != it_end; ++it, i++)
        centroids.push_back(Point((int)(*it)[colx], (int)(*it)[coly]));
}

void rransacUpdate(RRANSAC &rransac, Mat &Z, Mat &T, Mat &N,
    vector<Point> &centroids, vector<Feats> &feats, int frameCount)
{
    rransac.tracks.clear();

    int numMeasLost = rransac.set_dataInput(&Z, &T, &N,
                                            &centroids, rransac.Nw, frameCount);
    // cast vector features to a vector of BaseFeats shared pointers
    vector<shared_ptr<BaseFeats> > basefeats(feats.size());
    for (size_t f = 0; f < feats.size(); f++)
        basefeats[f] = std::make_shared<Feats>(feats[f]);

    rransac.apply(&T, &Z, &N, numMeasLost, basefeats);
    basefeats.clear();
}


vector<Track> getRRANSACendedTracks(RRANSAC &rransac)
{
    std::vector<Track> available(rransac.tracks.size());
    for (size_t t = 0; t < rransac.tracks.size(); t++)
    {
        shared_ptr<Feats> features = std::dynamic_pointer_cast<Feats>(rransac.tracks[t].features);
        if (features){
            available[t].position = vector<array<float, 2> >(rransac.tracks[t].trace.rows+1);
            available[t].frames = vector<int>(rransac.tracks[t].trace.rows+1); // == feats framesN
            // ler feats.framesN de trás para frente
            std::reverse(features->framesN.begin(), features->framesN.end());
            int r;
            for (r = 0; r < rransac.tracks[t].trace.rows; r++)
            {
                float * traceptr = (float *) rransac.tracks[t].trace.ptr<float>(r);
                available[t].position[r][0] = traceptr[0]; // x
                available[t].position[r][1] = traceptr[1]; // y
                available[t].frames[r] = (int) features->framesN[r]; // frameN
            }
            // o último dado fica em xhat, não vai para trace
            // r já recebeu o último ++, não precisa de +1
            available[t].position[r][0] =  rransac.tracks[t].xhat.at<float>(0, 0); // x
            available[t].position[r][1] =  rransac.tracks[t].xhat.at<float>(1, 0); // y
            available[t].frames[r] = (int) features->framesN[r+1]; // frameN
        }
    }
    return available;
}



// ---------------- Write to output yml
void writeParameters(FileStorage &fs, RRANSAC &rransac, Options &opt)
{
    fs << "M" << rransac.M;
    fs << "U" << rransac.U;
    fs << "tau_rho" << rransac.tau_rho;
    fs << "tau_T" << rransac.tau_T;
    fs << "Nw" << rransac.Nw;
    fs << "ell" << rransac.ell;
    fs << "tauR" << rransac.tauR;
    fs << "Qm" << opt.Qm;
    fs << "Rmx" << opt.Rmx;
    fs << "Rmy" << opt.Rmy;
}

void writeTrack2yml(Track & track, FileStorage &fs)
{
    fs << "{:";
    fs << "frames" << "[:";
    for (size_t i = 0; i < track.frames.size(); i++)
        fs << track.frames[i];
    fs << "]";
    fs << "position" << "[:";
    for (size_t i = 0; i < track.position.size(); i++)
        fs << "[:" << track.position[i][0] << track.position[i][1] << "]";
    fs << "]";
    fs << "}";
}




// --------- display results over video
void display(cv::Mat &frame, RRANSAC &rransac, std::vector<Point> &centroids)
{
    //frame.setTo(Scalar(255,255,255));
    for (Point p: centroids)
        circle(frame, p, 5, Scalar(255,255,255), -1, CV_AA);
    rransac.plotModels(&frame);
}



// void comoLer(char * filename){
//     printf("arquivo: %s", filename);
//     FileStorage fs(filename, FileStorage::READ);
//     FileNode features = fs["features"];
//     FileNodeIterator it = features.begin(), it_end = features.end();
//     int i = 0;
//     for (; it != it_end; ++it, i++) {
//         int frameN = (int)(*it)["frame"];
//         cout << "frame: " << frameN << "  centroids: [";
//         FileNode centroids = (*it)["centroids"];
//         FileNodeIterator cit = centroids.begin(), cit_end = centroids.end();
//         int nc = 0;
//         for (; cit != cit_end; ++cit, nc++) {
//             cout << "(" << (int)(*cit)[0] << "," << (int)(*cit)[1] << ")";
//         }
//         cout << "]" << endl;
//     }
//     fs.release();
// }
