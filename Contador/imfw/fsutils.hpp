#ifndef FSUTILS_HPP
#define FSUTILS_HPP

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

bool fs_exists(const fs::path& p, fs::file_status s = fs::file_status{})
{
    if(fs::status_known(s) ? fs::exists(s) : fs::exists(p))
        return true;
    else
        return false;
}

#endif