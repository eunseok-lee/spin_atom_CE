/*
 find_min.c
 written by Eunseok Lee
 
 function: find the id of minimum value element
 
 v1: Feb 2, 2018
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int find_min(double *a, int m, int n, int n_pivot)
{
    int i, id;
    double min_value;

    id = 0;
    min_value = *((a+id*n)+n_pivot);
    //printf("testarray[%d] = %f, min_value = %f\n",0,*((a+0)+n_pivot), min_value);
    for (i=1;i<m;i++) {
	//printf("testarray[%d] = %f, min_value = %f\n",i,*((a+i*n)+n_pivot), min_value);
	if ( *((a+i*n)+n_pivot) < min_value ) {
	    min_value = *((a+i*n)+n_pivot);
	    id = i;
	    //printf(" -> Event occured. min_value: %f, id = %d\n",min_value,id);
	}
    }
    //printf("min_id in subloop is %d\n",id);
    return(id);
}
