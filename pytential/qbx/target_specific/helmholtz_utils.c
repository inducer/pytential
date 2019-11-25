/* This file contains routines for evaluating spherical Bessel and Hankel
   functions.

   This is based on cdjseval3d.f and helmrouts3d.f from fmmlib3d, translated
   with a hacked version of f2c and manually postprocessed. */

/* Original copyright notice: */

/* **********************************************************************

Copyright (c) 2009-2012, Leslie Greengard, Zydrunas Gimbutas
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

********************************************************************** */

#include <complex.h>

/* declarations for f2c generated things */

typedef int integer;
typedef double doublereal;
typedef double complex doublecomplex;

static inline double z_abs(doublecomplex *z) {
    return cabs(*z);
}

static inline void z_exp(doublecomplex *out, doublecomplex *z) {
    *out = cexp(*z);
}

static inline void z_sin(doublecomplex *out, doublecomplex *z) {
    *out = csin(*z);
}

static inline void z_cos(doublecomplex *out, doublecomplex *z) {
    *out = ccos(*z);
}

/* Start of functions borrowed from cdjseval3d.f */

/*     Computation of spherical Bessel functions via recurrence */

/* ********************************************************************** */
/* Subroutine */ int jfuns3d_(integer *ier, integer *nterms, doublecomplex *
	z__, doublereal *scale, doublecomplex *fjs, integer *ifder,
	doublecomplex *fjder, integer *lwfjs, integer *iscale, integer *ntop)
{
    /* Initialized data */

    static doublereal upbound2 = 1e40;
    static doublereal upbound2inv = 1e-40;
    static doublereal tiny = 1e-200;
    static doublereal done = 1.;
    static doublereal zero = 0.;

    /* System generated locals */
    integer i__1;
    doublereal d__1, d__2;
    doublecomplex z__1;

    /* Local variables */
    integer i__;
    doublereal d0, d1, dd, dc1, dc2;
    doublecomplex fj0, fj1, zinv, ztmp;
    doublereal dcoef;
    doublereal sctot;
    doublecomplex zscale;
    doublereal scalinv;

/* ********************************************************************** */

/* PURPOSE: */

/* 	This subroutine evaluates the first NTERMS spherical Bessel */
/* 	functions and if required, their derivatives. */
/* 	It incorporates a scaling parameter SCALE so that */

/* 		fjs_n(z)=j_n(z)/SCALE^n */
/* 		fjder_n(z)=\frac{\partial fjs_n(z)}{\partial z} */

/* 	NOTE: The scaling parameter SCALE is meant to be used when */
/*             abs(z) < 1, in which case we recommend setting */
/* 	      SCALE = abs(z). This prevents the fjs_n from */
/*             underflowing too rapidly. */
/* 	      Otherwise, set SCALE=1. */
/* 	      Do not set SCALE = abs(z) if z could take on the */
/*             value zero. */
/*             In an FMM, when forming an expansion from a collection of */
/*             sources, set SCALE = min( abs(k*r), 1) */
/*             where k is the Helmholtz parameter and r is the box dimension */
/*             at the relevant level. */

/* INPUT: */

/*    nterms (integer): order of expansion of output array fjs */
/*    z     (complex *16): argument of the spherical Bessel functions */
/*    scale    (real *8) : scaling factor (discussed above) */
/*    ifder  (integer): flag indicating whether to calculate "fjder" */
/* 		          0	NO */
/* 		          1	YES */
/*    lwfjs  (integer): upper limit of input arrays */
/*                         fjs(0:lwfjs) and iscale(0:lwfjs) */
/*    iscale (integer): integer workspace used to keep track of */
/*                         internal scaling */

/* OUTPUT: */

/*    ier    (integer): error return code */
/*                         ier=0 normal return; */
/*                         ier=8 insufficient array dimension lwfjs */
/*    fjs   (complex *16): array of scaled Bessel functions. */
/*    fjder (complex *16): array of derivs of scaled Bessel functions. */
/*    ntop  (integer) : highest index in arrays fjs that is nonzero */

/*       NOTE, that fjs and fjder arrays must be at least (nterms+2) */
/*       complex *16 elements long. */




/* ... Initializing ... */

    *ier = 0;

/*       set to asymptotic values if argument is sufficiently small */

    if (z_abs(z__) < tiny) {
	fjs[0] = done;
	i__1 = *nterms;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    fjs[i__] = zero;
	}

	if (*ifder == 1) {
	    i__1 = *nterms;
	    for (i__ = 0; i__ <= i__1; ++i__) {
		fjder[i__] = zero;
	    }
	    fjder[1] = done / (*scale * 3);
	}

	return 0;
    }

/* ... Step 1: recursion up to find ntop, starting from nterms */

    *ntop = 0;
    zinv = done / *z__;
    fjs[*nterms] = done;
    fjs[*nterms - 1] = zero;

    i__1 = *lwfjs;
    for (i__ = *nterms; i__ <= i__1; ++i__) {
	dcoef = (i__ << 1) + done;
	ztmp = dcoef * zinv * fjs[i__] - fjs[i__ - 1];
	fjs[i__ + 1] = ztmp;

/* Computing 2nd power */
	d__1 = creal(ztmp);
/* Computing 2nd power */
	d__2 = cimag(ztmp);
	dd = d__1 * d__1 + d__2 * d__2;
	if (dd > upbound2) {
	    *ntop = i__ + 1;
	    break;
	}
    }
    if (*ntop == 0) {
	*ier = 8;
	return 0;
    }

/* ... Step 2: Recursion back down to generate the unscaled jfuns: */
/*             if magnitude exceeds UPBOUND2, rescale and continue the */
/* 	      recursion (saving the order at which rescaling occurred */
/* 	      in array iscale. */

    i__1 = *ntop;
    for (i__ = 0; i__ <= i__1; ++i__) {
	iscale[i__] = 0;
    }

    fjs[*ntop] = zero;
    fjs[*ntop - 1] = done;
    for (i__ = *ntop - 1; i__ >= 1; --i__) {
	dcoef = (i__ << 1) + done;
	ztmp = dcoef * zinv * fjs[i__] - fjs[i__ + 1];
	fjs[i__ - 1] = ztmp;

/* Computing 2nd power */
	d__1 = creal(ztmp);
/* Computing 2nd power */
	d__2 = cimag(ztmp);
	dd = d__1 * d__1 + d__2 * d__2;
	if (dd > upbound2) {
	    fjs[i__] *= upbound2inv;
	    fjs[i__ - 1] *= upbound2inv;
	    iscale[i__] = 1;
	}
/* L2200: */
    }

/* ...  Step 3: go back up to the top and make sure that all */
/*              Bessel functions are scaled by the same factor */
/*              (i.e. the net total of times rescaling was invoked */
/*              on the way down in the previous loop). */
/*              At the same time, add scaling to fjs array. */

    scalinv = done / *scale;
    sctot = 1.;
    i__1 = *ntop;
    for (i__ = 1; i__ <= i__1; ++i__) {
	sctot *= scalinv;
	if (iscale[i__ - 1] == 1) {
	    sctot *= upbound2inv;
	}
	fjs[i__] *= sctot;
    }

/* ... Determine the normalization parameter: */

    z_sin(&z__1, z__);
    fj0 = z__1 * zinv;
    z_cos(&z__1, z__);
    fj1 = fj0 * zinv - z__1 * zinv;

    d0 = z_abs(&fj0);
    d1 = z_abs(&fj1);
    if (d1 > d0) {
	zscale = fj1 / (fjs[1] * *scale);
    } else {
	zscale = fj0 / fjs[0];
    }

/* ... Scale the jfuns by zscale: */

    ztmp = zscale;
    i__1 = *nterms;
    for (i__ = 0; i__ <= i__1; ++i__) {
	fjs[i__] *= ztmp;
    }

/* ... Finally, calculate the derivatives if desired: */

    if (*ifder == 1) {
	fjs[*nterms + 1] *= ztmp;

	fjder[0] = -fjs[1] * *scale;
	i__1 = *nterms;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    dc1 = i__ / ((i__ << 1) + done);
	    dc2 = done - dc1;
	    dc1 *= scalinv;
	    dc2 *= *scale;
	    fjder[i__] = dc1 * fjs[i__ - 1] - dc2 * fjs[i__ + 1];
	}
    }
    return 0;
} /* jfuns3d_ */

/* Start of functions borrowed from helmrouts3d.f */

/*      This file contains the basic subroutines for */
/*      forming and evaluating multipole (partial wave) expansions. */

/*      Documentation is incomplete and ongoing... */


/*      Remarks on scaling conventions. */

/*      1)  Hankel and Bessel functions are consistently scaled as */
/*       	hvec(n)= h_n(z)*scale^(n) */
/*       	jvec(n)= j_n(z)/scale^(n) */

/*          In some earlier FMM implementations, the convention */
/*       	hvec(n)= h_n(z)*scale^(n+1) */
/*          was sometimes used, leading to various obscure rescaling */
/*          steps. */

/*          scale should be of the order of |z| if |z| < 1. Otherwise, */
/*          scale should be set to 1. */


/*      2) There are many definitions of the spherical harmonics, */
/*         which differ in terms of normalization constants. We */
/*         adopt the following convention: */

/*         For m>0, we define Y_n^m according to */

/*         Y_n^m = \sqrt{2n+1} \sqrt{\frac{ (n-m)!}{(n+m)!}} \cdot */
/*                 P_n^m(\cos \theta)  e^{i m phi} */
/*         and */

/*         Y_n^-m = dconjg( Y_n^m ) */

/*         We omit the Condon-Shortley phase factor (-1)^m in the */
/*         definition of Y_n^m for m<0. (This is standard in several */
/*         communities.) */

/*         We also omit the factor \sqrt{\frac{1}{4 \pi}}, so that */
/*         the Y_n^m are orthogonal on the unit sphere but not */
/*         orthonormal.  (This is also standard in several communities.) */
/*         More precisely, */

/*                 \int_S Y_n^m Y_n^m d\Omega = 4 \pi. */

/*         Using our standard definition, the addition theorem takes */
/*         the simple form */

/*         e^( i k r}/(ikr) = */
/*         \sum_n \sum_m  j_n(k|S|) Ylm*(S) h_n(k|T|) Ylm(T) */


/* ----------------------------------------------------------------------- */
/*      h3d01: computes h0, h1 (first two spherical Hankel fns.) */
/*      h3dall: computes Hankel functions of all orders and scales them */
/* ********************************************************************** */
/* Subroutine */ static int h3d01_(doublecomplex *z__, doublecomplex *h0,
	doublecomplex *h1)
{
    /* Initialized data */

    static doublecomplex eye = I;
    static doublereal thresh = 1e-15;
    static doublereal done = 1.;

    /* System generated locals */
    doublecomplex z__1;

    /* Local variables */
    doublecomplex cd;

/* ********************************************************************** */

/*     Compute spherical Hankel functions of order 0 and 1 */

/*     h0(z)  =   exp(i*z)/(i*z), */
/*     h1(z)  =   - h0' = -h0*(i-1/z) = h0*(1/z-i) */

/* ----------------------------------------------------------------------- */
/*     INPUT: */

/* 	z   :  argument of Hankel functions */
/*              if abs(z)<1.0d-15, returns zero. */

/* ----------------------------------------------------------------------- */
/*     OUTPUT: */

/* 	h0  :  h0(z)    (spherical Hankel function of order 0). */
/* 	h1  :  -h0'(z)  (spherical Hankel function of order 1). */

/* ----------------------------------------------------------------------- */

    if (z_abs(z__) < thresh) {
	*h0 = 0.;
	*h1 = 0.;
	return 0;
    }

/*     Otherwise, use formula */

    cd = eye * *z__;
    z_exp(&z__1, &cd);
    *h0 = z__1 / cd;
    *h1 = *h0 * (done / *z__ - eye);

    return 0;
} /* h3d01_ */




/* ********************************************************************** */
/* Subroutine */ int h3dall_(integer *nterms, doublecomplex *z__, doublereal *
	scale, doublecomplex *hvec, integer *ifder, doublecomplex *hder)
{
    /* Initialized data */
    static doublereal thresh = 1e-15;
    static doublereal done = 1.;

    /* Builtin functions */
    double z_abs(doublecomplex *);

    /* Local variables */
    integer i__;
    integer i__1;
    doublereal dtmp;
    doublecomplex zinv, ztmp;
    doublereal scal2;

/* ********************************************************************** */

/*     This subroutine computes scaled versions of the spherical Hankel */
/*     functions h_n of orders 0 to nterms. */

/*       	hvec(n)= h_n(z)*scale^(n) */

/*     The parameter SCALE is useful when |z| < 1, in which case */
/*     it damps out the rapid growth of h_n as n increases. In such */
/*     cases, we recommend setting */

/*               scale = |z| */

/*     or something close. If |z| > 1, set scale = 1. */

/*     If the flag IFDER is set to one, it also computes the */
/*     derivatives of h_n. */

/* 		hder(n)= h_n'(z)*scale^(n) */

/*     NOTE: If |z| < 1.0d-15, the subroutine returns zero. */

/* ----------------------------------------------------------------------- */
/*     INPUT: */

/*     nterms  : highest order of the Hankel functions to be computed. */
/*     z       : argument of the Hankel functions. */
/*     scale   : scaling parameter discussed above */
/*     ifder   : flag indcating whether derivatives should be computed. */
/* 		ifder = 1   ==> compute */
/* 		ifder = 0   ==> do not compute */

/* ----------------------------------------------------------------------- */
/*     OUTPUT: */

/*     hvec    : the vector of spherical Hankel functions */
/*     hder    : the derivatives of the spherical Hankel functions */

/* ----------------------------------------------------------------------- */


/*     If |z| < thresh, return zeros. */

    if (z_abs(z__) < thresh) {
	i__1 = *nterms;
	for (i__ = 0; i__ <= i__1; ++i__) {
	    hvec[i__] = 0;
	    hder[i__] = 0;
	}
	return 0;
    }

/*     Otherwise, get h_0 and h_1 analytically and the rest via */
/*     recursion. */

    h3d01_(z__, hvec, &hvec[1]);
    hvec[0] = hvec[0];
    hvec[1] *= *scale;

/*     From Abramowitz and Stegun (10.1.19) */

/*     h_{n+1}(z)=(2n+1)/z * h_n(z) - h_{n-1}(z) */

/*     With scaling: */

/*     hvec(n+1)=scale*(2n+1)/z * hvec(n) -(scale**2) hvec(n-1) */

    scal2 = *scale * *scale;
    zinv = *scale / *z__;
    i__1 = *nterms - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dtmp = (i__ << 1) + done;
	ztmp = zinv * dtmp;
	hvec[i__ + 1] = ztmp * hvec[i__] - scal2 * hvec[i__ - 1];
    }

/*     From Abramowitz and Stegun (10.1.21) */

/* 	h_{n}'(z)= h_{n-1}(z) - (n+1)/z * h_n(z) */

/*     With scaling: */

/*     hder(n)=scale* hvec(n-1) - (n+1)/z * hvec(n) */


    if (*ifder == 1) {

	hder[0] = -hvec[1] / *scale;
	zinv = 1. / *z__;
	i__1 = *nterms;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    dtmp = i__ + done;
	    ztmp = zinv * dtmp;
	    hder[i__] = *scale * hvec[i__ - 1] - ztmp * hvec[i__];
	}
    }

    return 0;
} /* h3dall_ */
