/* f2c.h  --  Standard Fortran to C header file */

/**  barf  [ba:rf]  2.  "He suggested using FORTRAN, and everybody barfed."

	- From The Shogakukan DICTIONARY OF NEW ENGLISH (Second edition) */

#ifndef F2C_INCLUDE
#define F2C_INCLUDE

#include <complex.h>

typedef int integer;
typedef unsigned int uinteger;
typedef double doublereal;
typedef double complex doublecomplex;

static inline double z_abs(doublecomplex *z) {
  return cabs(*z);
}

static inline void z_sin(doublecomplex *out, doublecomplex *z) {
  *out = csin(*z);
}

static inline void z_cos(doublecomplex *out, doublecomplex *z) {
  *out = ccos(*z);
}

#endif
