/*
 obtainerr2_par.c
 written by Eunseok Lee
 
 function: solve a linear equation system and obtain the squared error
 
 v1: Feb 2, 2018
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <cem.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

double obtainerr2_par(int row_ini, int row_end, double *matA, int n_row, int n_col, int *col_ids, int n_col_ids, double *y) {
    
    int i, j, k, tmp;
    double err[n_row];
    double sum;
    
    gsl_matrix * A = gsl_matrix_alloc (n_row-1,n_col_ids);
    gsl_matrix * V = gsl_matrix_alloc (n_col_ids,n_col_ids);
    gsl_vector * S = gsl_vector_alloc (n_col_ids);
    gsl_vector * work = gsl_vector_alloc (n_col_ids);
//    gsl_matrix * U = gsl_matrix_alloc (n_row-1,n_col_ids);
    gsl_vector * b = gsl_vector_alloc (n_row-1);
    gsl_vector * x = gsl_vector_alloc (n_col_ids);
    
    for (i=0;i<n_row;i++)
        err[i] = 0.0;
    
    // need to copy y to b!
    for (i=row_ini;i<row_end;i++) {
        for (j=0;j<i;j++) {
            for (k=0; k<n_col_ids; k++) {
                tmp = *(col_ids+k);
                gsl_matrix_set (A, j, k, *(matA+j*n_col+tmp));
            }
            gsl_vector_set (b, j, y[j]);
        }
        for (j=i+1;j<n_row;j++) {
            for (k=0; k<n_col_ids; k++) {
                tmp = *(col_ids+k);
                gsl_matrix_set (A, j-1, k, *(matA+j*n_col+tmp));
            }
            gsl_vector_set (b, j-1, y[j]);
        }

        gsl_linalg_SV_decomp(A, V, S, work);    // on output, A is replaced by U!
        gsl_linalg_SV_solve (A, V, S, b, x);    // So, A should be used instead of U, here.
        
        sum = 0.0;
        for (k=0;k<n_col_ids;k++) {
            sum += *(matA+i*n_col+col_ids[k]) * gsl_vector_get(x,k);
        }
        err[i] = y[i] - sum;
    }
    
    sum = 0.0;
    for (i=row_ini;i<row_end;i++)
        sum += err[i]*err[i];
    
    gsl_matrix_free(A);
    gsl_matrix_free(V);
    gsl_vector_free(S);
    gsl_vector_free(work);
    gsl_vector_free(b);
    gsl_vector_free(x);

    return(sum);
}
