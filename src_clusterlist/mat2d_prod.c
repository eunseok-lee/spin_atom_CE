/*
 mat2d_prod.c
 by Eunseok Lee
 
 function: calculae the product of two 2-d matrix
 
 v1: Feb 2, 2018
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void mat2d_prod(double *c, int mc, int nc, double *a, int ma, int na, double *b, int mb, int nb) 
{
    // matrix production c=a*b
    int i, j, k;
    double aik, bkj, cij;
    for (i=0;i<mc;i++)
	for (j=0;j<nc;j++) {
	    cij = 0.0;
	    for (k=0;k<nb;k++) {
		aik = *((a+i*na)+k);
		bkj = *((b+k*nb)+j);
		cij = cij + aik*bkj; 
	    }
	    *((c+i*nc)+j) = cij;
	}
}
