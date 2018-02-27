/*
 find_recipro.c
 by Eunseok Lee

 function: map to reciprocal space w.r.t. 0
 
 v1: Feb 2, 2018
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double find_recipro(double x)
{
    //printf("received %g, ",x);
    if (x>0) {
	//printf("return %g\n",x-1);
	return(x-1);
    }
    else if (x<0) {
	//printf("return %g\n",x+1);
	return(x+1);
    }
    else {
	//printf("return %g\n",0);
	return(0.0);
    }
}

