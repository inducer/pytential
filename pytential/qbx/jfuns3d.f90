!! Copyright (C) 2009-2012: Leslie Greengard and Zydrunas Gimbutas
!! Contact: greengard@cims.nyu.edu
!!
!! This software is being released under a modified FreeBSD license
!! (see COPYING in home directory).
!**********************************************************************

!    $Date: 2011-07-15 16:28:31 -0400 (Fri, 15 Jul 2011) $
!    $Revision: 2253 $


!     Computation of spherical Bessel functions via recurrence

!**********************************************************************
    subroutine jfuns3d(ier,nterms,z,scale,fjs,ifder,fjder, &
    lwfjs,iscale,ntop)
    implicit none
    integer :: ier,nterms,ifder,lwfjs,ntop,i,ncntr
    real *8 :: scale,d0,d1,dc1,dc2,dcoef,dd,done,tiny,zero
    real *8 :: scalinv,sctot,upbound,upbound2,upbound2inv
!**********************************************************************

! PURPOSE:

!	This subroutine evaluates the first NTERMS spherical Bessel
!	functions and if required, their derivatives.
!	It incorporates a scaling parameter SCALE so that

!		fjs_n(z)=j_n(z)/SCALE^n
!		fjder_n(z)=\frac{\partial fjs_n(z)}{\partial z}

!	NOTE: The scaling parameter SCALE is meant to be used when
!             abs(z) < 1, in which case we recommend setting
!	      SCALE = abs(z). This prevents the fjs_n from
!             underflowing too rapidly.
!	      Otherwise, set SCALE=1.
!	      Do not set SCALE = abs(z) if z could take on the
!             value zero.
!             In an FMM, when forming an expansion from a collection of
!             sources, set SCALE = min( abs(k*r), 1)
!             where k is the Helmholtz parameter and r is the box dimension
!             at the relevant level.

! INPUT:

!    nterms (integer): order of expansion of output array fjs
!    z     (complex *16): argument of the spherical Bessel functions
!    scale    (real *8) : scaling factor (discussed above)
!    ifder  (integer): flag indicating whether to calculate "fjder"
!		          0	NO
!		          1	YES
!    lwfjs  (integer): upper limit of input arrays
!                         fjs(0:lwfjs) and iscale(0:lwfjs)
!    iscale (integer): integer workspace used to keep track of
!                         internal scaling

! OUTPUT:

!    ier    (integer): error return code
!                         ier=0 normal return;
!                         ier=8 insufficient array dimension lwfjs
!    fjs   (complex *16): array of scaled Bessel functions.
!    fjder (complex *16): array of derivs of scaled Bessel functions.
!    ntop  (integer) : highest index in arrays fjs that is nonzero

!       NOTE, that fjs and fjder arrays must be at least (nterms+2)
!       complex *16 elements long.


    integer :: iscale(0:lwfjs)
    complex *16 :: wavek,fjs(0:lwfjs),fjder(0:*)
    complex *16 :: z,zinv,com,fj0,fj1,zscale,ztmp

    data upbound/1.0d+32/, upbound2/1.0d+40/, upbound2inv/1.0d-40/
    data tiny/1.0d-200/,done/1.0d0/,zero/0.0d0/

! ... Initializing ...

    ier=0

!       set to asymptotic values if argument is sufficiently small

    if (abs(z) < tiny) then
        fjs(0) = done
        do i = 1, nterms
            fjs(i) = zero
        enddo

        if (ifder == 1) then
            do i=0,nterms
                fjder(i)=zero
            enddo
            fjder(1)=done/(3*scale)
        endif

        RETURN
    endif

! ... Step 1: recursion up to find ntop, starting from nterms

    ntop=0
    zinv=done/z
    fjs(nterms)=done
    fjs(nterms-1)=zero

    do i=nterms,lwfjs
        dcoef=2*i+done
        ztmp=dcoef*zinv*fjs(i)-fjs(i-1)
        fjs(i+1)=ztmp

        dd = dreal(ztmp)**2 + dimag(ztmp)**2
        if (dd > upbound2) then
            ntop=i+1
            exit
        endif
    enddo
    if (ntop == 0) then
        ier=8
        return
    endif

! ... Step 2: Recursion back down to generate the unscaled jfuns:
!             if magnitude exceeds UPBOUND2, rescale and continue the
!	      recursion (saving the order at which rescaling occurred
!	      in array iscale.

    do i=0,ntop
        iscale(i)=0
    enddo

    fjs(ntop)=zero
    fjs(ntop-1)=done
    do i=ntop-1,1,-1
        dcoef=2*i+done
        ztmp=dcoef*zinv*fjs(i)-fjs(i+1)
        fjs(i-1)=ztmp

        dd = dreal(ztmp)**2 + dimag(ztmp)**2
        if (dd > UPBOUND2) then
            fjs(i) = fjs(i)*UPBOUND2inv
            fjs(i-1) = fjs(i-1)*UPBOUND2inv
            iscale(i) = 1
        endif
    enddo

! ...  Step 3: go back up to the top and make sure that all
!              Bessel functions are scaled by the same factor
!              (i.e. the net total of times rescaling was invoked
!              on the way down in the previous loop).
!              At the same time, add scaling to fjs array.

    ncntr=0
    scalinv=done/scale
    sctot = 1.0d0
    do i=1,ntop
        sctot = sctot*scalinv
        if(iscale(i-1) == 1) sctot=sctot*UPBOUND2inv
        fjs(i)=fjs(i)*sctot
    enddo

! ... Determine the normalization parameter:

    fj0=sin(z)*zinv
    fj1=fj0*zinv-cos(z)*zinv

    d0=abs(fj0)
    d1=abs(fj1)
    if (d1 > d0) then
        zscale=fj1/(fjs(1)*scale)
    else
        zscale=fj0/fjs(0)
    endif

! ... Scale the jfuns by zscale:

    ztmp=zscale
    do i=0,nterms
        fjs(i)=fjs(i)*ztmp
    enddo

! ... Finally, calculate the derivatives if desired:

    if (ifder == 1) then
        fjs(nterms+1)=fjs(nterms+1)*ztmp

        fjder(0)=-fjs(1)*scale
        do i=1,nterms
            dc1=i/(2*i+done)
            dc2=done-dc1
            dc1=dc1*scalinv
            dc2=dc2*scale
            fjder(i)=dc1*fjs(i-1)-dc2*fjs(i+1)
        enddo
    endif
    return
    end subroutine jfuns3d

    ! void c_jfuns3(
    !     int *ier,
    !     int nterms,
    !     complex double z,
    !     double scale,
    !     complex double *fjs,
    !     int ifder,
    !     complex double *fjder,
    !     int lwfjs,
    !     int *iscale,
    !     int *ntop)
    subroutine c_jfuns3d(ier, nterms, z, scale, fjs, ifder, fjder, lwfjs,&
         & iscale, ntop) bind(c)
      use iso_c_binding
      implicit none
      integer (c_int) :: ier
      integer (c_int), value :: nterms
      complex (c_double_complex), value :: z
      real (c_double), value :: scale
      complex (c_double_complex) :: fjs(0:lwfjs)
      integer (c_int), value :: ifder
      complex (c_double_complex) :: fjder(0:*)
      integer (c_int), value :: lwfjs
      integer (c_int) :: iscale(0:lwfjs)
      integer (c_int) :: ntop

      call jfuns3d(ier, nterms, z, scale, fjs, ifder, fjder, lwfjs, iscale,&
           & ntop)
    end subroutine c_jfuns3d
