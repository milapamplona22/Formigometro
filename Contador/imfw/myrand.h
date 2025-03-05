#pragma once
#include "stdio.h"

#ifndef UCHAR_MAX
#define UCHAR_MAX 255
#endif


inline unsigned _time_seed() {
    time_t now = time(0);
    unsigned char *p = (unsigned char *)&now;
    unsigned seed = 0;
    size_t i;

    for ( i = 0; i < sizeof now; i++ )
        seed = seed * ( UCHAR_MAX + 2U ) + p[i];

    return seed;
}

inline void setMyRand(){
    srand(_time_seed());
}

inline double _uniform_deviate ( int seed ) {
    return seed * ( 1.0 / ( RAND_MAX + 1.0 ) );
}

inline double _unirand ( double a, double b ) {
    //return ( a+ (int) ((b-a+1)*((double)rand()/(double)RAND_MAX)) );
    return  a + _uniform_deviate ( rand() ) * ( b - a +1 );
}

template <typename T>
T myrand_uniform( T a, T b ){
    return static_cast<T>( _unirand( (double) a, (double) b ) );
}

