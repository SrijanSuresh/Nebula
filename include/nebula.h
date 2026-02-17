#ifndef NEBULA_H
#define NEBULA_H

#ifdef __cplusplus
extern "C" {
#endif

void launch_nebula_kernel(float* d_pos, float time, int numStars);

#ifdef __cplusplus
}
#endif

#endif // NEBULA_H
