/*
 eval_anomaly.c
 written by Eunseok Lee
 
 function: assess the anomaly of each data set
 
 v1. Feb 2, 2018
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <mpi.h>
//#include <cem.h>

//void sort_array_ids(double*, int*, int);

void eval_anomaly(double *measurement, double *instance, int ins_row, int ins_col, double *label, int n_label_types, int knn) {
    
    int i, j, k;
    double tmp_sum;
    double knn_avg_dist;
    
//    double measurement[ins_row];
    double dij2[ins_row];
    int ids[ins_row];
    
    for (i=0;i<ins_row;i++) {
        measurement[i] = 0.0;
        for (j=0;j<ins_row;j++) {
            dij2[i] = 0.0;
            for (k=0;k<ins_col;k++) {
                dij2[j] = dij2[i] + pow(*(instance+ins_col*j+k) - *(instance+ins_col*i+k),2.0);
            }
            ids[j] = j;
        }
        sort_array_ids(dij2,ids,ins_row);
        tmp_sum = 0;
        for (j=0;j<knn;j++)
            tmp_sum = tmp_sum + sqrt(dij2[j]);
        knn_avg_dist = tmp_sum*1.0/knn;
        tmp_sum = 0.0;
        for (j=0;j<knn;j++)
            for (k=0;k<n_label_types;k++)
                tmp_sum = tmp_sum + pow(*(label + n_label_types*ids[j]+k) - *(label + n_label_types*i+k),2.0)/knn/n_label_types;
        measurement[i] = sqrt(tmp_sum*1.0)/ins_col;
    }
    
//    for (i=0;i<ins_row;i++)
//        printf("anomaly_score[%d] = %f\n",i,measurement[i]);
}
    
