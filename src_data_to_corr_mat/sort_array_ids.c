/*
 sort_array.c
 by Eunseok Lee

 function: sort a double array

 v1: Feb 2, 2018
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void sort_array_ids(double *a, int *b, int n) {

    int i, j, swap_id;;
    double swap;
    
    for (i=0;i<n;i++)
        for (j=0;j<(n-i-1);j++)
            if (*(a+j) > *(a+j+1)) {
                swap = *(a+j);
                *(a+j) = *(a+j+1);
                *(a+j+1) = swap;
                swap_id = b[j];
                b[j] = b[j+1];
                b[j+1] = swap_id;
            }
}









