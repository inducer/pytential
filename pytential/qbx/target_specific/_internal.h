#ifndef UTILS_H
#define UTILS_H

#include <complex.h>

extern int jfuns3d_(int *ier, int *nterms, double complex *z,
                    double *scale, double complex *fjs, int *ifder,
                    double complex *fjder, int *lwfjs, int *iscale,
                    int *ntop);

extern int h3dall_(int *nterms, double complex *z, double *scale,
		   double complex *hvec, int *ifder, double complex *hder);

#endif  // UTILS_H
