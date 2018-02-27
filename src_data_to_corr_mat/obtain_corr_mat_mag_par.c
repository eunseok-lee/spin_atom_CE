/*
 obtain_corr_mat_mag_par.c
 written by Eunseok Lee
 
 function: calculate the correlation matrix for data-magmom configurations
 
 v1. Feb 2, 2018
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <cem.h>

void obtain_corr_mat_mag_par(int row_ini, int row_end, double *corr_mat, int ndata, int ncorr_col, int *data, int *magmom, int c2start, int c3start, int *nlist, int *map_to_cluster2, int *map_to_cluster3, int np, int neighbor_num) {
    
    int i, j, k, l, m, n, nj, nk, tmp, tmp1;
    double chevy1a[ndata][np], chevy2a[ndata][np];
    double chevy1s[ndata][np], chevy2s[ndata][np];
    int ij_array[2], ijk_array[3];
    double phia[4], phis[4];
    
    // initialize the corr_mat matrix
    for (i=0;i<ndata;i++)
        for (j=0;j<ncorr_col;j++)
            *(corr_mat+ncorr_col*i+j) = 0.0;
    printf("corr_mat calculation initialization, Done, size: %d*%d \n",ndata,ncorr_col);
    // empty cluster & point cluster
    for (i=0;i<ndata;i++)
        for (j=0;j<np;j++) {
            tmp = *(data+np*i+j);
            chevy1a[i][j] = sqrt(3.0/2)*tmp;
            chevy2a[i][j] = sqrt(2.0)*(1-3.0/2*tmp*tmp);
            tmp = *(magmom+np*i+j);
            chevy1s[i][j] = sqrt(3.0/2)*tmp;
            chevy2s[i][j] = sqrt(2.0)*(1-3.0/2*tmp*tmp);
        }
    for (i=0;i<ndata;i++) {
        *(corr_mat+ncorr_col*i+0) = 1.0;
        for (j=row_ini;j<row_end;j++) {
            *(corr_mat+ncorr_col*i+1) += chevy1a[i][j]*chevy1s[i][j];
            *(corr_mat+ncorr_col*i+2) += chevy1a[i][j]*chevy2s[i][j];
            *(corr_mat+ncorr_col*i+3) += chevy2a[i][j]*chevy1s[i][j];
            *(corr_mat+ncorr_col*i+4) += chevy2a[i][j]*chevy2s[i][j];
        }
    }
    printf("corr_mat_trial calculation: empty and point clusters, Done\n");

    // pair clusters
    for (i=row_ini;i<row_end;i++)
        for (nj=0;nj<neighbor_num;nj++) {
            j = *(nlist+neighbor_num*i+nj);
            ij_array[0] = i;
            ij_array[1] = j;
            sort_int_array(ij_array,2);
            if (i==ij_array[0] && j==ij_array[1]) {
                for (l=0;l<ndata;l++) {
                    tmp1 = *(map_to_cluster2+neighbor_num*i+nj);
                    if (tmp1 >= 0) {
                        tmp = 9*tmp1 + c2start;
                        phia[0] = chevy1a[l][i]*chevy1a[l][j];
                        phia[1] = 0.5*(chevy1a[l][i]*chevy2a[l][j]+chevy1a[l][j]*chevy2a[l][i]);
                        phia[2] = chevy2a[l][i]*chevy2a[l][j];
                        phis[0] = chevy1s[l][i]*chevy1s[l][j];
                        phis[1] = 0.5*(chevy1s[l][i]*chevy2s[l][j]+chevy1s[l][j]*chevy2s[l][i]);
                        phis[2] = chevy2s[l][i]*chevy2s[l][j];
                        for (m=0;m<3;m++)
                            for (n=0;n<3;n++) {
                                *(corr_mat+ncorr_col*l+tmp+3*m+n) += phia[m]*phis[n];
                            }
                    }
                }
            }
        }
    printf("corr_mat_trial calculation: pair clusters, Done\n");

    // three-body clusters
    for (i=row_ini;i<row_end;i++)
        for (nj=0;nj<neighbor_num;nj++) {
            j = *(nlist+neighbor_num*i+nj);
            for (nk=0;nk<neighbor_num;nk++) {
                k = *(nlist+neighbor_num*i+nk);
                ijk_array[0] = i;
                ijk_array[1] = j;
                ijk_array[2] = k;
                sort_int_array(ijk_array,3);
                if (i==ijk_array[0] && j==ijk_array[1] && k==ijk_array[2]) {
                    for (l=0;l<ndata;l++) {
                        tmp1 = *(map_to_cluster3+neighbor_num*i+nj+np*neighbor_num*nk);
                        if (tmp1 >= 0) {
//                            printf("Error in three-body cluster mapping: map3(%d,%d,%d) -> %d\n",i,nj,nk,tmp1);
                            tmp = 16*tmp1 + c3start;
                            phia[0] = chevy1a[l][i]*chevy1a[l][j]*chevy1a[l][k];
                            phia[1] = (chevy1a[l][i]*chevy1a[l][j]*chevy2a[l][k]+chevy1a[l][i]*chevy2a[l][j]*chevy1a[l][k]+chevy2a[l][i]*chevy1a[l][j]*chevy1a[l][k])/3.0;
                            phia[2] = (chevy1a[l][i]*chevy2a[l][j]*chevy2a[l][k]+chevy2a[l][i]*chevy1a[l][j]*chevy2a[l][k]+chevy2a[l][i]*chevy2a[l][j]*chevy1a[l][k])/3.0;
                            phia[3] = chevy2a[l][i]*chevy2a[l][j]*chevy2a[l][k];
                            phis[0] = chevy1s[l][i]*chevy1s[l][j]*chevy1s[l][k];
                            phis[1] = (chevy1s[l][i]*chevy1s[l][j]*chevy2s[l][k]+chevy1s[l][i]*chevy2s[l][j]*chevy1s[l][k]+chevy2s[l][i]*chevy1s[l][j]*chevy1s[l][k])/3.0;
                            phis[2] = (chevy1s[l][i]*chevy2s[l][j]*chevy2s[l][k]+chevy2s[l][i]*chevy1s[l][j]*chevy2s[l][k]+chevy2s[l][i]*chevy2s[l][j]*chevy1s[l][k])/3.0;
                            phis[3] = chevy2s[l][i]*chevy2s[l][j]*chevy2s[l][k];
                            for (m=0;m<4;m++)
                                for (n=0;n<4;n++) {
                                    *(corr_mat+ncorr_col*l+tmp+4*m+n) += phia[m]*phis[n];
                                }
                        }
                    }
                }
            }
        }
//    for (i=0;i<ncorr_col;i++)
//        printf("%f\n",*(corr_mat+i));
    printf("corr_mat_trial calculation: three-body clusters, Done\n");
}













