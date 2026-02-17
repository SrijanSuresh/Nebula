#ifndef NEBULA_H
#define NEBULA_H

// This tells the C++ compiler: "Don't mangle these names, look for them like C functions"
extern "C" {
    void launch_nebula_kernel(float* d_pos, float time, int numStars);
}

#endif
