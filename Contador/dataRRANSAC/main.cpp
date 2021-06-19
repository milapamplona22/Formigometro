#include <iostream>
#include "opencv2/opencv.hpp"
#include <array>
#include <numeric> //std::iota
#include "rransacFrameN.hpp"
// ../imfw
#include "argParse.hpp"
// ../imfw
#include "video.hpp"


using namespace cv;
using namespace std;


struct Track
{
    vector<array<float, 2> > position;
    vector<int> frames;
};

struct Options
{
    string infs;
    string outfs;
    string outfwfs; bool has_framewiseyml;
    string displayVid; bool has_video;
    // RRANSAC params
    int M, Nw, ell, tau_T, tauR;
    float U, tau_rho, Qm, Rmx, Rmy;
};

// ---------------------------- Options
void menu(char * name) {
    printf ("  %s -data <data yml> -out <tracks yml> [RRANSAC params]\n\n"
            "  -data   \t name of yml file containing centroids and frames\n"
            "  -out    \t output yml file, will have tracks positions and frames\n"
            "  -outfw  \t framewise output yml file, will have tracks positions and frames\n"
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
            "  -Rmy    \t (float) Ry multiplier\n\n", name);
}
bool parseOptions(Options &opt, int argc, char ** argv);
bool rransacOptions(RRANSAC &rransac, Options &opt, int argc, char **argv);
void printParameters(RRANSAC &rransac, Options &opt);

// ----------------- read from input yml
inline Size readSize(const cv::FileNode &fn);
template <typename T>
    vector<T> readVec(const cv::FileNode &fn);


// ----------- setting data for RRANSAC
//             input and getting output
void getCentroids(const FileNode &fn, vector<Point> &centroids, int colx=0, int coly=1);
void rransacUpdate(RRANSAC &rransac, Mat &Z, Mat &T, Mat &N,
    vector<Point> &centroids, int frameCount);
vector<Track> getRRANSACendedTracks(RRANSAC &rransac);


// ---------------- Write to output yml
void writeParameters(FileStorage &fs, RRANSAC &rransac, Options &opt);
void writeTrack2yml(Track & track, FileStorage &fs);
vector<tuple<int, vector<int>, vector<array<float, 2> > > >
    tracksFramewise(vector<Track> & tracks);
template<typename T>
void reorder(std::vector<T>& vec, std::vector<size_t> order);
void tracksFramewise2(vector<Track> & tracks, vector<int> & frames, vector<vector<int> > & ids,
    vector<vector<array<float, 2> > > & positions);
void writeTrack2ymlFramewise(
    vector<tuple<int, vector<int>, vector<array<float, 2> > > > & tracksFw,
    FileStorage &fs);
void writeTrack2ymlFramewise2(vector<int> & frames, vector<vector<int> > & ids,
    vector<vector<array<float, 2> > > & positions, FileStorage &fs);

// --------- display results over video
void display(cv::Mat &frame, RRANSAC &rransac, std::vector<Point> &centroids);



int main(int argc, char ** argv)
{
    // Parse arguments
    Options opt;
    if (!parseOptions(opt, argc, argv)){
        menu(argv[0]);
        return 0;
    }
    
    // Parse and set RRANSAC
    Mat Z, T, N;
    RRANSAC rransac;
    if (!rransacOptions(rransac, opt, argc, argv)){
        menu(argv[0]);
        return 0;
    }
    //printParameters(rransac, opt);

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

    FileStorage outfwfs;
    if (opt.has_framewiseyml){
        outfwfs = FileStorage(opt.outfwfs, FileStorage::WRITE);
        outfwfs << "input" << opt.infs;
        outfwfs << "size" << size;
        writeParameters(outfwfs, rransac, opt); // write rransac parameters used
    }

    // if display over video
    Video video;
    Mat frame(size, CV_8UC3);
    if (opt.has_video)
        video = Video(opt.displayVid);


    int totalTracks = 0;
    int frameN = -1;
    vector<Track> alltracks;
    for (int i = 0; it != it_end; ++it, i++)
    {
        frameN = (int)(*it)["frame"]; // read frame n
        // read centroids 
        vector<Point> centroids;
        getCentroids((*it)["centroids"], centroids, 0, 1);
        
        // create feats (frameN) for each centroid seen
        // update RRANSAC
        rransacUpdate(rransac, Z, T, N, centroids, frameN);
    
        // get finished tracks
        vector<Track> tracks = getRRANSACendedTracks(rransac);
        //printf ("\n\n");
        //export tracks to outfile
        for (size_t t = 0; t < tracks.size(); t++) {
            writeTrack2yml(tracks[t], outfs);
            totalTracks++;
        }
        
        if (opt.has_framewiseyml)
            alltracks.insert(alltracks.end(), tracks.begin(), tracks.end());
        
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
    
    if (opt.has_framewiseyml) {
        outfwfs << "tracks" << (int)alltracks.size();
        vector<int> frames;
        vector<vector<int> > ids;
        vector<vector<array<float, 2> > > positions;
        tracksFramewise2(alltracks, frames, ids, positions);
        writeTrack2ymlFramewise2(frames, ids, positions, outfwfs);
        //auto fwt = tracksFramewise(alltracks);
        //writeTrack2ymlFramewise(fwt, outfwfs);
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
    bool out = parseArg(argc, argv, "-out", opt.outfs);
    opt.has_framewiseyml = parseArg(argc, argv, "-outfw", opt.outfwfs);
    if (!out && !opt.has_framewiseyml) {
        complete = false;
        printf("option -out (or -outfw) missing. There has to be one of these output options\n");
    }
    opt.has_video = parseArg(argc, argv, "-display", opt.displayVid);
    return complete;
}

bool rransacOptions(RRANSAC &rransac, Options &opt, int argc, char **argv)
{
    bool complete = true;
    if (!parseArg(argc, argv, "-M", opt.M)){ printf ("option -M missing\n"); complete = false; }
    if (!parseArg(argc, argv, "-U", opt.U)){ printf ("option -U missing\n"); complete = false; }
    if (!parseArg(argc, argv, "-tau_rho", opt.tau_rho)){ printf ("option -tau_rho missing\n"); complete = false; }
    if (!parseArg(argc, argv, "-tau_T", opt.tau_T)){ printf ("option -tau_T missing\n"); complete = false; }
    if (!parseArg(argc, argv, "-Nw", opt.Nw)){ printf ("option -Nw missing\n"); complete = false; }
    if (!parseArg(argc, argv, "-ell", opt.ell)){ printf ("option -ell missing\n"); complete = false; }
    if (!parseArg(argc, argv, "-tauR", opt.tauR)){ printf ("option -tauR missing\n"); complete = false; }
    if (!parseArg(argc, argv, "-Qm", opt.Qm)){ printf ("option -Qm missing\n"); complete = false; }
    if (!parseArg(argc, argv, "-Rmx", opt.Rmx)){ printf ("option -Rmx missing\n"); complete = false; }
    if (!parseArg(argc, argv, "-Rmy", opt.Rmy)){ printf ("option -Rmy missing\n"); complete = false; }
    if (!complete)
        return false;

    rransac.setup(
        opt.M,        // M: Number of stored models
        opt.U,        // U: Merge threshold
        opt.tau_rho,  // tau_rho: Good model threshold
        opt.tau_T,    // tau_T: Minimum number of time steps needed for a good model
        opt.Nw,       // Nw: Measurement window size
        opt.ell,      // ell: Number of RANSAC iterations
        opt.tauR,     // tauR: Inlier threshold (multiplier)
        opt.Qm,       // Qm
        opt.Rmx,      // Rmx
        opt.Rmy       // Rmy
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
    vector<Point> &centroids, int frameCount)
{
    rransac.tracks.clear();
    int n_centroids = (int) centroids.size();
    int numMeasLost = rransac.set_dataInput(&Z, &T, &N,
                                            &centroids, rransac.Nw, frameCount);
    
    rransac.apply(&T, &Z, &N, numMeasLost, n_centroids, frameCount);
}


vector<Track> getRRANSACendedTracks(RRANSAC &rransac)
{
    std::vector<Track> available(rransac.tracks.size());
    for (size_t t = 0; t < rransac.tracks.size(); t++)
    {
        available[t].frames = rransac.tracks[t].framesN;
        available[t].position = vector<array<float, 2> >(rransac.tracks[t].trace.rows+1);
        int r;
        for (r = 0; r < rransac.tracks[t].trace.rows; r++)
        {
            float * traceptr = (float *) rransac.tracks[t].trace.ptr<float>(r);
            available[t].position[r][0] = traceptr[0]; // x
            available[t].position[r][1] = traceptr[1]; // y
        }
        // o último dado fica em xhat, não vai para trace
        // r já recebeu o último ++, não precisafw de +1
        available[t].position[r][0] =  rransac.tracks[t].xhat.at<float>(0, 0); // x
        available[t].position[r][1] =  rransac.tracks[t].xhat.at<float>(1, 0); // y
    }
    return available;
}



// ---------------- fwWrfwite to output yml
void writeParameters(FileStorage &fs, RRANSAC &rransac, Options &opt)
{
    fs << "M" << rransac.M;
    fs << "U" << rransac.U;
    fs << "tau_rho" << rransac.tau_rho;
    fs << "tau_T" << rransac.tau_T;
    fs << "Nw" << rransac.Nw;
    fs << "ell" << rransac.ell;
    fs << "tauR" << opt.tauR; // o rransac.tauR é modificado em rransac::setup
    fs << "Qm" << opt.Qm;
    fs << "Rmx" << opt.Rmx;
    fs << "Rmy" << opt.Rmy;
}


void printParameters(RRANSAC &rransac, Options &opt)
{
    std::cout << "M = " << rransac.M <<std::endl; 
    std::cout << "U = " << rransac.U <<std::endl; 
    std::cout << "tau_rho = " << rransac.tau_rho <<std::endl; 
    std::cout << "tau_T = " << rransac.tau_T <<std::endl; 
    std::cout << "Nw = " << rransac.Nw <<std::endl; 
    std::cout << "ell = " << rransac.ell <<std::endl; 
    std::cout << "tauR = " << opt.tauR <<std::endl; // o rransac.tauR é modificado em rransac::setup
    std::cout << "Qm = " << opt.Qm <<std::endl; 
    std::cout << "Rmx = " << opt.Rmx <<std::endl; 
    std::cout << "Rmy = " << opt.Rmy <<std::endl; 
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

template<typename T>
void reorder(std::vector<T>& vec, std::vector<size_t> order)
{
    // apply order in vec
    for( size_t iv = 0; iv < vec.size() - 1; ++iv ){
            if (order[iv] == iv)
                continue;
            size_t io;
            for(io = iv + 1; io < order.size(); ++io){
                if (order[io] == iv)
                    break;
            }
            std::swap( vec[iv], vec[order[iv]] );
            std::swap( order[iv], order[io] );
    }
}

void tracksFramewise2(vector<Track> & tracks, vector<int> & frames,
    vector<vector<int> > & ids, vector<vector<array<float, 2> > > & positions)
{
    // vector of  <frame: track ids, respectivefw tracks pofwsitions>
    vector<int>::iterator idx;
    for (size_t t = 0; t < tracks.size(); t++){
        for (size_t f = 0; f < tracks[t].frames.size(); f++){
            int fnum = tracks[t].frames[f];
            idx = std::find_if(frames.begin(), frames.end(),
                [&fnum](const int f) {
                           return f == fnum;
            });
            int i = idx - frames.begin();
            if (idx == frames.end()){
                frames.push_back(fnum);
                ids.push_back(vector<int>({(int)t}));
                positions.push_back(vector<array<float, 2> >({tracks[t].position[f]}));
            }
            else {
                ids[i].push_back((int)t);
                positions[i].push_back(tracks[t].position[f]);
            }
        }
    }
    // reordenar vetores de acordo com frames
    std::vector<size_t> fids(frames.size());
    std::iota(fids.begin(), fids.end(), 0);
    std::sort(fids.begin(), fids.end(),
        [&frames](int A, int B) -> bool { return frames[A] < frames[B]; });
    reorder(frames, fids);
    reorder(ids, fids);
    reorder(positions, fids);
}

void writeTrack2ymlFramewise2( vector<int> & frames, vector<vector<int> > & ids,
    vector<vector<array<float, 2> > > & positions, FileStorage &fs)
{
    fs << "frames" << "[:"; // open tracks section to write to
    for (size_t i = 0; i < frames.size(); i++)
        fs << frames[i];
    fs << "]";
    fs << "ids" << "[:"; // open tracks section to write to
    for (size_t i = 0; i < ids.size(); i++) {
        fs << "[:";
        for (size_t j = 0; j < ids[i].size(); j++)
            fs << ids[i][j];
        fs << "]";
    }
    fs << "]";
    fs << "positions" << "[:";
    for (size_t i = 0; i < positions.size(); i++) {
        fs << "[:";
        for (size_t j = 0; j < positions[i].size(); j++)
            fs << "[:" << positions[i][j][0] << positions[i][j][1] << "]";
        fs << "]";
    }
    fs << "]";

}


vector<tuple<int, vector<int>, vector<array<float, 2> > > >
    tracksFramewise(vector<Track> & tracks)
{
    // vector of  <frame: track ids, respectivefw tracks pofwsitions>
    vector<tuple<int, vector<int>, vector<array<float, 2> > > > data;
    vector<tuple<int, vector<int>, vector<array<float, 2> > > >::iterator idx;
    for (int t = 0; t < (int)tracks.size(); t++){
        for (int f = 0; f < (int)tracks[t].frames.size(); f++){
            int fnum = tracks[t].frames[f];
            idx = std::find_if(data.begin(), data.end(), 
                [&fnum](const tuple<int, vector<int>, vector<array<float, 2> > > & t) {
                           return std::get<0>(t) == fnum;
            });
            if (idx == data.end())
                data.push_back(make_tuple(
                    fnum,
                    vector<int>({t}),
                    vector<array<float, 2> >({tracks[t].position[f]})
                    ));
            else {
                std::get<1>(*idx).push_back(t);
                std::get<2>(*idx).push_back(tracks[t].position[f]);
            }
        }
    }
    std::sort(data.begin(), data.end(),
        [](const tuple<int, vector<int>, vector<array<float, 2> > > & t1,
            const tuple<int, vector<int>, vector<array<float, 2> > > & t2) {
                           return std::get<0>(t1) < std::get<0>(t2);
            });

    return data;
}

void writeTrack2ymlFramewise(
    vector<tuple<int, vector<int>, vector<array<float, 2> > > > & tracksFw,
    FileStorage &fs)
{
    fs << "frames" << "["; // open tracks section to write to
    for (size_t i = 0; i < tracksFw.size(); i++) {
        fs << "{:";
        fs << "frame" << std::get<0>(tracksFw[i]);
        fs << "id" << "[:";
        for (size_t j = 0; j < std::get<1>(tracksFw[i]).size(); j++)
            fs << std::get<1>(tracksFw[i])[j];
        fs << "]";
        fs << "position" << "[:";
        for (size_t j = 0; j < std::get<2>(tracksFw[i]).size(); j++)
            fs << "[:" << std::get<2>(tracksFw[i])[j][0] << std::get<2>(tracksFw[i])[j][1] << "]";
        fs << "]";
        fs<< "}";
    }
    fs << "]"; // close frames
}



// --------- display results over video
void display(cv::Mat &frame, RRANSAC &rransac, std::vector<Point> &centroids)
{
    //frame.setTo(Scalar(255,255,255));
    for (Point p: centroids)
        circle(frame, p, 5, Scalar(255,255,255), -1, cv::LINE_AA);
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
