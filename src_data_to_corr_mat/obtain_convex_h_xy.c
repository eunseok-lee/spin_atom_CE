/*
 obtain_convex_h_xy.c
 written by Eunseok Lee
 
 function: calculate the convex hull of a function y = f(x)
 
 v1. Feb 2, 2018
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void obtain_convex_h_xy(double *cvh, int n_bin, int *x, double *y, int n_data) {
    
    int i, j, k, nid, id1, id2, ctr;
    int x1;
    double y1;
    int convex_h_ids[n_data];
    int convex_h_x[n_data];
    double convex_h_y[n_data];

    printf("Convex hull is created.\n");
/*    printf("Input data\n");
    printf("---------------------------\n");
    for (i=0;i<n_data;i++)
        printf("%d %f\n",x[i],y[i]);
    printf("\n");
*/
    // locate the left bottom
    x1 = x[0];
    y1 = y[0];
    id1 = 0;
    for (i=1;i<n_data;i++)
        if (x[i] <= x1 && y[i] < y1) {
            id1 = i;
            x1 = x[i];
            y1 = y[i];
        }
    nid = 0;
    convex_h_ids[nid] = id1;
    convex_h_x[nid] = x[id1];
    convex_h_y[nid] = y[id1];
    
    printf("The left-bottom point is id:%d-(%d, %f)\n",id1,x[id1],y[id1]);
    
    // start the convex hull formulation
    ctr = 1;
    while (ctr == 1) {
        // for a new convex hull point
        id2 = id1 + 1;
        for (i=0;i<n_data;i++) {
            if (i==id1)
                continue;
            if ((y[id2]-y[id1])*(x[i]-x[id1])-(x[id2]-x[id1])*(y[i]-y[id1]) > 0) {
                id2 = i;
            }
        }
        if (x[id2] <= x[id1])
            ctr = 0;
        else {
            nid++;
            convex_h_ids[nid] = id2;
            convex_h_x[nid] = x[id2];
            convex_h_y[nid] = y[id2];
            id1 = id2;
        }
    }
    
    printf("%d cvh points were identified.\n",nid+1);
    for (i=0;i<=nid;i++)
        printf("%d %d %f\n",convex_h_ids[i],convex_h_x[i],convex_h_y[i]);
    printf("\n");
    
    for (i=0;i<n_bin;i++)
        for (j=0;j<nid;j++)
            if (i >= convex_h_x[j] && i <= convex_h_x[j+1]) {
                cvh[i] = (convex_h_y[j+1]-convex_h_y[j])*1.0/(convex_h_x[j+1]-convex_h_x[j])*(i-convex_h_x[j]) + convex_h_y[j];
//                printf("cvh[%d] = %f\n",i,cvh[i]);
                break;
            }
    
}













