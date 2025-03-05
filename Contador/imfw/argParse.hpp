#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <sstream>


template <typename T> T convertTo (const char * value) {
    std::string str(value);
    std::istringstream ss(str);
    T num;
    ss >> num;
    return num;
}


template <typename T>
bool parseArg(int argc, char ** argv, std::string arg,
    std::vector<T> &values) {
    
    bool isThere = false;
    bool found = false;
    values.clear();
    for (int i = 1; i < argc; i++) {
        if (argv[i]) {
            std::string s_argvi(argv[i]);
            if (found) {
                if (argv[i][0] == '-')
                    found = false;
                else
                    values.push_back(convertTo<T>(argv[i]));
            }
            if (s_argvi == arg) {
                isThere = true;
                found = true;
            }
        }
    }
    return isThere;
}

template <typename T>
bool parseArg(int argc, char ** argv, std::string arg,
    T &value) {
    
    for (int i = 1; i < argc; i++) {
        if (argv[i]) {
            std::string s_argvi(argv[i]);
            if (s_argvi == arg){
                value = convertTo<T>(argv[i+1]);
                return true;
            }
        }
    }
    return false;    
}

bool parseArg(int argc, char ** argv, std::string arg) {
    for (int i = 1; i < argc; i++) {
        if (argv[i]) {
            std::string s_argvi(argv[i]);
            if (s_argvi == arg)
                return true;
        }
    }
    return false;    
}
