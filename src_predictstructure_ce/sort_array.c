#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void sort_array(double *a, int n) 
{  
    int i, j;
    double swap;

    for (i=0;i<n;i++)
	for (j=0;j<(n-i-1);j++)
	    if (*(a+j) > *(a+j+1)) {
		swap = *(a+j);
		*(a+j) = *(a+j+1);
		*(a+j+1) = swap;
	    }
}

