/* Based on helmrouts3d.f from fmmlib3d, translated with modified f2c */

#include "f2c.h"

/* Original copyright notice: */

/* ************************************************************************** */
/* Copyright (C) 2009-2012: Leslie Greengard and Zydrunas Gimbutas */
/* Contact: greengard@cims.nyu.edu */
/* */
/* This software is being released under a modified FreeBSD license */
/* (see COPYING in home directory). */
/* ************************************************************************** */

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
/* Subroutine */ int h3d01_(doublecomplex *z__, doublecomplex *h0, 
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

