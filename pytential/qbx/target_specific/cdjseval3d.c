/* cdjseval3d.f -- translated by f2c (version 20160102).
New version
*/

#include "f2c.h"

/* c Copyright (C) 2009-2012: Leslie Greengard and Zydrunas Gimbutas */
/* c Contact: greengard@cims.nyu.edu */
/* c */
/* c This software is being released under a modified FreeBSD license */
/* c (see COPYING in home directory). */
/* cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc */

/*    $Date: 2011-07-15 16:28:31 -0400 (Fri, 15 Jul 2011) $ */
/*    $Revision: 2253 $ */


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

