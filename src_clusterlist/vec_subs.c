/*
 vec_subs.c
 by Eunseok Lee
 
 function: calculate the substration between two 3 element vectors
 
 v1: Feb 2, 2018
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void vec_subs(double *c, double *a, double *b)
{
    int i;
    *(c+0) = *(a+0) - *(b+0);
    *(c+1) = *(a+1) - *(b+1);
    *(c+2) = *(a+2) - *(b+2);
}
