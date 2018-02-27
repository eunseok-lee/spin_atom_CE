/*
 sort_int_array.c
 written by Eunseok Lee
 
 function: sort an integer array
 
 v1. Feb 2, 2018
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <cem.h>

void sort_int_array(int *a, int n)
{
    int i, j;
    int swap;
    
    for (i=0;i<n;i++)
        for (j=0;j<(n-i-1);j++)
            if (*(a+j) > *(a+j+1)) {
                swap = *(a+j);
                *(a+j) = *(a+j+1);
                *(a+j+1) = swap;
            }
}



















