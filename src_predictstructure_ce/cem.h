#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// clusterlist.c
extern void mat2d_prod(double*,int,int,double*,int,int,double*,int,int);
extern void vec_subs(double*,double*,double*);
extern double fnorm(double*);
extern int find_min(double*,int,int,int);
extern void sort_array(double*,int);
extern double find_recipro(double);

extern int np;
extern int ncluster1;
extern int ncluster2;
extern int ncluster3;

extern int *map_to_cluster2;
extern int *map_to_cluster3;

// findcluster_x.c
extern void load_double_array(double*,char);
extern void load_int_array(int*,char);
extern int mat_size(int*);
extern void obtain_corr_mat(double*, int, int, int*, int, int);

// predictstructure_MC_ternary.c
extern void randperm(int*, int);
extern void obtain_corr_mat(int*, int*, int, int);
extern void mat_copy(double*, double*, int, int, int);

// obtain_corr_mat.c
extern double mat2d_sum_row(double*, int , int);
