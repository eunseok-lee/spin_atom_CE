/*
 clusterlist.c
 by Eunseok Lee
 
 function: formulate the clusters and cluster functions
 
 v1: Feb 2, 2018
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
//#include <cem.h>

void mat2d_prod(double*,int,int,double*,int,int,double*,int,int);
void vec_subs(double*,double*,double*);
double fnorm(double*);
int find_min(double*,int,int,int);
void sort_array_int(int*,int);
void sort_array(double*,int);
double find_recipro(double);
void load_double_mat(double*, int, int, int, char*);

int main(int argc, char **argv)
{
    int n1, n2, n3;
    int nbasis;
    int np;
    double kv[3][3], pk[3][3];
    double pa1, pa2, pa3;
    double pb1, pb2, pb3;
    double pc1, pc2, pc3;
    double rp_tmp[3];
    double *rp1, *rp2, *rp3;
    double *rpL1, *rpL2, *rpL3;
    double *rpN1, *rpN2, *rpN3;
    double *rpO1, *rpO2, *rpO3;
    double *rpT1, *rpT2, *rpT3;
    double neighbor_dist_max;
    int neighbor_num_max=100;   //default value of neighbor_num_max
    int neighbor_num;

    int i, j, k, l, m, n;
    
    FILE *fp;
    char paramfilename[100]="param.dat";
    char basisfilename[100];
    fp = fopen(paramfilename,"r");
    char buff_line[200], dummy[200], param_name[100];
    int param_name_len;
    while(fgets(buff_line,sizeof(buff_line),fp) != NULL) {
        if (buff_line[0] == '#') {
            //            printf("Comment line: %s",buff_line);
            continue;
        }
        strcpy(param_name,"pa");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %lf %lf %lf",dummy,&pa1,&pa2,&pa3);
            printf("pa = %f %f %f\n",pa1,pa2,pa3);
        }
        strcpy(param_name,"pb");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %lf %lf %lf",dummy,&pb1,&pb2,&pb3);
            printf("pb = %f %f %f\n",pb1,pb2,pb3);
        }
        strcpy(param_name,"pc");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %lf %lf %lf",dummy,&pc1,&pc2,&pc3);
            printf("pc = %f %f %f\n",pc1,pc2,pc3);
        }
        strcpy(param_name,"basisfilename");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %s",dummy,basisfilename);
            printf("basisfilename = %s\n",basisfilename);
        }
        strcpy(param_name,"nbasis");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&nbasis);
            printf("nbasis = %d\n",nbasis);
        }
        strcpy(param_name,"n1");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&n1);
            printf("n1 = %d\n",n1);
        }
        strcpy(param_name,"n2");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&n2);
            printf("n2 = %d\n",n2);
        }
        strcpy(param_name,"n3");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&n3);
            printf("n3 = %d\n",n3);
        }
        strcpy(param_name,"neighbor_num_max");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&neighbor_num_max);
            printf("neighbor_num_max = %d\n",neighbor_num_max);
        }
        strcpy(param_name,"neighbor_dist_max");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %lf",dummy,&neighbor_dist_max);
            printf("neighbor_dist_max = %f\n",neighbor_dist_max);
        }
    }

    pk[0][0] = pa1; pk[0][1] = pa2; pk[0][2] = pa3;
    pk[1][0] = pb1; pk[1][1] = pb2; pk[1][2] = pb3;
    pk[2][0] = pc1; pk[2][1] = pc2; pk[2][2] = pc3;
    
    double pp[nbasis][3];
    double *basis_tmp;
    basis_tmp = (double*) malloc(nbasis*3*sizeof(double));
    load_double_mat(basis_tmp,nbasis,3,1,basisfilename);
    for (i=0;i<nbasis;i++) {
        pp[i][0] = *(basis_tmp + 3*i + 0);
        pp[i][1] = *(basis_tmp + 3*i + 1);
        pp[i][2] = *(basis_tmp + 3*i + 2);
        printf("pp[%d][:] = %f, %f, %f\n",i,pp[i][0],pp[i][1],pp[i][2]);
    }
    
/*    pp[0][0] = 0.0; pp[0][1] = 0.0; pp[0][2] = 0.0;
    pp[1][0] = 0.5; pp[1][1] = 0.5; pp[1][2] = 0.0;
    pp[2][0] = 0.0; pp[2][1] = 0.5; pp[2][2] = 0.5;
    pp[3][0] = 0.5; pp[3][1] = 0.0; pp[3][2] = 0.5;
    pp[4][0] = 0.0; pp[4][1] = 0.75; pp[4][2] = 0.73;
    pp[5][0] = 0.5; pp[5][1] = 0.75; pp[5][2] = 0.23;
    pp[6][0] = 0.0; pp[6][1] = 0.25; pp[6][2] = 0.23;
    pp[7][0] = 0.5; pp[7][1] = 0.25; pp[7][2] = 0.73;
*/
    kv[0][0] = pk[0][0]*n1; kv[0][1] = pk[0][1]*n1; kv[0][2] = pk[0][2]*n1;
    kv[1][0] = pk[1][0]*n2; kv[1][1] = pk[1][1]*n2; kv[1][2] = pk[1][2]*n2;
    kv[2][0] = pk[2][0]*n3; kv[2][1] = pk[2][1]*n3; kv[2][2] = pk[2][2]*n3;

    rp1 = (double*)malloc(n1*n2*n3*4*sizeof(double));
    rp2 = (double*)malloc(n1*n2*n3*4*sizeof(double));
    rp3 = (double*)malloc(n1*n2*n3*4*sizeof(double));
    rpL1 = (double*)malloc(n1*n2*n3*2*sizeof(double));
    rpL2 = (double*)malloc(n1*n2*n3*2*sizeof(double));
    rpL3 = (double*)malloc(n1*n2*n3*2*sizeof(double));
    rpN1 = (double*)malloc(n1*n2*n3*2*sizeof(double));
    rpN2 = (double*)malloc(n1*n2*n3*2*sizeof(double));
    rpN3 = (double*)malloc(n1*n2*n3*2*sizeof(double));
    rpO1 = (double*)malloc(n1*n2*n3*4*sizeof(double));
    rpO2 = (double*)malloc(n1*n2*n3*4*sizeof(double));
    rpO3 = (double*)malloc(n1*n2*n3*4*sizeof(double));
    m = 0;
    for (i=0; i<n1; i++)
        for (j=0; j<n2; j++)
            for (k=0; k<n3; k++) {
                for (l=0; l<4; l++) {
                    *(rp1+4*m+l) = (i+pp[l][0])/n1;
                    *(rp2+4*m+l) = (j+pp[l][1])/n2;
                    *(rp3+4*m+l) = (k+pp[l][2])/n3;
		}
                for (l=0; l<2; l++) {
                    *(rpL1+2*m+l) = (i+pp[l][0])/n1;
                    *(rpL2+2*m+l) = (j+pp[l][1])/n2;
                    *(rpL3+2*m+l) = (k+pp[l][2])/n3;
                }
                for (l=0; l<2; l++) {
                    *(rpN1+2*m+l) = (i+pp[l+2][0])/n1;
                    *(rpN2+2*m+l) = (j+pp[l+2][1])/n2;
                    *(rpN3+2*m+l) = (k+pp[l+2][2])/n3;
                }
                for (l=0; l<4; l++) {
                    *(rpO1+4*m+l) = (i+pp[l+4][0])/n1;
                    *(rpO2+4*m+l) = (j+pp[l+4][1])/n2;
                    *(rpO3+4*m+l) = (k+pp[l+4][2])/n3;
                }
                m++;
            }
    np = 4*m;
    printf("The number of lattice site: %d\n", np);

    neighbor_num = neighbor_num_max;
    int nlist_col_id;
    int nlist[np][neighbor_num];
    double nlist_dist[np][neighbor_num];
    int map_to_cluster1[np];
    int map_to_cluster2[np][neighbor_num];
//    int map_to_cluster3[np][neighbor_num][neighbor_num];
    int *map_to_cluster3;   //This will shrinks after cluster2 creation, not in the code but in printed data
    
    int cluster0 = 1;
    //    int map_to_cluster0 = 0;

    int cluster1 = 1;
    for (i=0;i<np;i++)
        map_to_cluster1[i] = 0;     //map to 1st cluster1 (all cluster1 are same by symmetry)
    int ncluster1 = 1;
    
    // pair clusters
    double *cluster2;
    int *store_pos2;
//    int map_to_cluster2[np][np];
    double dji[3], Dji[3];
    double dji_case[8][3], dji_case_kv[8][3];
    double dc[8];
    int ctr;
    int ncluster2;
    double dc_min;
    int dc_min_idx;
    
    cluster2 = (double*)malloc(500000*sizeof(double));
    store_pos2 = (int*)malloc(500000*sizeof(int));
    for (i=0;i<np;i++)
        for (j=0;j<neighbor_num;j++)
            map_to_cluster2[i][j] = -1;
    n = 0;
    for (i=0;i<np;i++) {
        nlist_col_id = -1;
        for (j=0;j<np;j++) {
            if (j == i)
                continue;
            
            dji[0] = *(rp1+j) - *(rp1+i);
            dji[1] = *(rp2+j) - *(rp2+i);
            dji[2] = *(rp3+j) - *(rp3+i);
            Dji[0] = find_recipro(dji[0]);
            Dji[1] = find_recipro(dji[1]);
            Dji[2] = find_recipro(dji[2]);
            dji_case[0][0] = dji[0]; dji_case[0][1] = dji[1]; dji_case[0][2] = dji[2];
            dji_case[1][0] = dji[0]; dji_case[1][1] = dji[1]; dji_case[1][2] = Dji[2];
            dji_case[2][0] = dji[0]; dji_case[2][1] = Dji[1]; dji_case[2][2] = dji[2];
            dji_case[3][0] = dji[0]; dji_case[3][1] = Dji[1]; dji_case[3][2] = Dji[2];
            dji_case[4][0] = Dji[0]; dji_case[4][1] = dji[1]; dji_case[4][2] = dji[2];
            dji_case[5][0] = Dji[0]; dji_case[5][1] = dji[1]; dji_case[5][2] = Dji[2];
            dji_case[6][0] = Dji[0]; dji_case[6][1] = Dji[1]; dji_case[6][2] = dji[2];
            dji_case[7][0] = Dji[0]; dji_case[7][1] = Dji[1]; dji_case[7][2] = Dji[2];
        
            // determine the bond length
            mat2d_prod(dji_case_kv,8,3,dji_case,8,3,kv,3,3);
            dc_min = 1e8;
            dc_min_idx = 100;
            for (k=0;k<8;k++) {
                dc[k] = sqrt(dji_case_kv[k][0]*dji_case_kv[k][0] + dji_case_kv[k][1]*dji_case_kv[k][1] + dji_case_kv[k][2]*dji_case_kv[k][2]);
                if (fmin(dc_min,dc[k]) == dc[k]) {
                    dc_min = dc[k];
                    dc_min_idx = k;
                }
            }
 
            // check if the bond length is within neighbor distance
            if (dc_min > neighbor_dist_max)
                continue;

            nlist_col_id++;
            nlist[i][nlist_col_id] = j;
            nlist_dist[i][nlist_col_id] = dc_min;
            
            // special case of n=0
            if (n==0) {
                *(cluster2+4*n+0) = dc_min;
                *(cluster2+4*n+1) = fabs(dji_case_kv[dc_min_idx][0]);
                *(cluster2+4*n+2) = fabs(dji_case_kv[dc_min_idx][1]);
                *(cluster2+4*n+3) = fabs(dji_case_kv[dc_min_idx][2]);
                *(store_pos2+2*n+0) = (int) fmin(i,j);
                *(store_pos2+2*n+1) = (int) fmax(i,j);
                map_to_cluster2[i][nlist_col_id] = n;
                n++;
                continue;
            }
            
            // look up the cluster list and check if current cluster in included.
            ctr = 0;
            for (k=0;k<n;k++)
                if (fabs(*(cluster2+4*k+0)-dc_min) < 0.1 && fabs(*(cluster2+4*k+3)-fabs(dji_case_kv[dc_min_idx][2])) < 0.1) {
                    ctr = 1;
                    map_to_cluster2[i][nlist_col_id] = k;
                  break;
                }
            if (ctr == 0) {
                *(cluster2+4*n+0) = dc_min;
                *(cluster2+4*n+1) = fabs(dji_case_kv[dc_min_idx][0]);
                *(cluster2+4*n+2) = fabs(dji_case_kv[dc_min_idx][1]);
                *(cluster2+4*n+3) = fabs(dji_case_kv[dc_min_idx][2]);
                *(store_pos2+2*n+0) = (int) fmin(i,j);
                *(store_pos2+2*n+1) = (int) fmax(i,j);
                map_to_cluster2[i][nlist_col_id] = n;
                n++;
            }
        }
    }
    ncluster2 = n;

    // Let's assume nlist_col_id is the same for all ion, bc. symmetry 
    neighbor_num = nlist_col_id+1;
    printf("neighbor_num = %d\n",neighbor_num);
    // Resize the matrixes
//    realloc(nlist,n*neighbor_num*sizeof(int));
//    realloc(map_to_cluster2,n*neighbor_num*size(int));
    printf("%d pair clusters were created.\n",ncluster2);
    
    // three-body clusters
    int jj, kk, min_idx, ncluster3;
    double *cluster3;
    int *store_pos3;
    double dki[3], Dki[3];
    double dki_case[8][3], dki_case_kv[8][3];
    double dji_vec[3], dki_vec[3], dkj_vec[3];
    double ac[3];
    double dji_norm, dki_norm, dkj_norm;
    double d_cluster3_ac[3];
    int neighbor1, neighbor2;
    int nj, nk;
    
    cluster3 = (double*)malloc(500000*sizeof(double));
    store_pos3 = (int*)malloc(500000*sizeof(int));
    map_to_cluster3 = (int*)malloc(np*neighbor_num*neighbor_num*sizeof(int));

    for (i=0;i<np;i++)
        for (j=0;j<neighbor_num;j++)
            for (k=0;k<neighbor_num;k++)
                *(map_to_cluster3+np*neighbor_num*k+neighbor_num*i+j) = -1;

    n = 0;
    for (i=0;i<np;i++)
        for (nj=0;nj<neighbor_num;nj++) {
            j = nlist[i][nj];
            for (nk=nj+1;nk<neighbor_num;nk++) {
                k = nlist[i][nk];
                ctr = 0;
                for (l=0;l<neighbor_num;l++)
                    if (nlist[j][l] == k) {
                        ctr = 1;
                        break;
                    }
                if (ctr == 0) {
                    *(map_to_cluster3+np*neighbor_num*nk+neighbor_num*i+nj) = -1;
                    continue;
                }
                ac[0] = (double) map_to_cluster2[i][nj]*1.0;
                ac[1] = (double) map_to_cluster2[i][nk]*1.0;
                ac[2] = (double) map_to_cluster2[j][l]*1.0;
  //              printf("i=%d,nj=%d,nk=%d,j=%d,k=%d,map[i][nj]=%d,map[i][nk]=%d,map[j][l]=%d\n",i,nj,nk,j,k,ac[0],ac[1],ac[2]);
//                sort_array_int(ac,3);
                sort_array(ac,3);
//                printf("after sorting: ac = %d, %d, %d",ac[0],ac[1],ac[2]);
                if (n==0) {
                    *(cluster3+3*n+0) = ac[0]; // *store ac[][1] not ac[][0]
                    *(cluster3+3*n+1) = ac[1];
                    *(cluster3+3*n+2) = ac[2];
                    *(store_pos3+3*n+0) = i;
                    *(store_pos3+3*n+1) = j;
                    *(store_pos3+3*n+2) = k;
                    *(map_to_cluster3+np*neighbor_num*nk+neighbor_num*i+nj) = n;
                    //		    printf("store_pos3[%d][]=%d,%d,%d\n",n,*(store_pos3+3*n+0),*(store_pos3+3*n+1),*(store_pos3+3*n+2));
                    n++;
                    continue;
                }
                
                // look up the cluster list and check if current cluster in included.
                ctr = 0;
                for (l=0;l<n;l++) {
                    if ( *(cluster3+3*l+0)==ac[0] &&  *(cluster3+3*l+1)==ac[1] &&  *(cluster3+3*l+2)==ac[2] ) {
                        ctr = 1;
                        *(map_to_cluster3+np*neighbor_num*nk+neighbor_num*i+nj) = l;
                        break;
                    }
                }
                if (ctr == 0) {
                    *(cluster3+3*n+0) = ac[0]; // *store ac[][1] not ac[][0]
                    *(cluster3+3*n+1) = ac[1];
                    *(cluster3+3*n+2) = ac[2];
                    *(store_pos3+3*n+0) = i;
                    *(store_pos3+3*n+1) = j;
                    *(store_pos3+3*n+2) = k;
                    *(map_to_cluster3+np*neighbor_num*nk+neighbor_num*i+nj) = n;
                    //		    printf("store_pos3[%d][]=%d,%d,%d\n",n,*(store_pos3+3*n+0),*(store_pos3+3*n+1),*(store_pos3+3*n+2));
                    n++;
                }
            }
        }
    ncluster3 = n;
    printf("%d three-body clusters were created.\n",ncluster3);

    // save the cluster lists

    FILE *fp1, *fp2, *fp3, *fp4, *fp5, *fp6, *fp7, *fp8, *fp9, *fp10, *fp11, *fp12;
    char dirname[100];
    char filename[200];
    sprintf(dirname,"dir_result");
    struct stat st = {0};
    if (stat(dirname, &st) == -1) {
        mkdir(dirname,0777);
        printf("Created a new directory, dir_result, for produced data storage.\n");
    }
    else
        printf("Updated the produced data in dir_result.\n");
    sprintf(filename,"%s/cluster2list.dat",dirname);
    fp1 = fopen(filename,"w");
    sprintf(filename,"%s/cluster3list.dat",dirname);
    fp2 = fopen(filename,"w");
    sprintf(filename,"%s/store_pos2.dat",dirname);
    fp3 = fopen(filename,"w");
    sprintf(filename,"%s/store_pos3.dat",dirname);
    fp4 = fopen(filename,"w");
    sprintf(filename,"%s/map_to_cluster2.dat",dirname);
    fp5 = fopen(filename,"w");
    sprintf(filename,"%s/map_to_cluster3.dat",dirname);
    fp6 = fopen(filename,"w");
    sprintf(filename,"%s/map_to_cluster1.dat",dirname);
    fp7 = fopen(filename,"w");
    sprintf(filename,"%s/nlist.dat",dirname);
    fp8 = fopen(filename,"w");
    sprintf(filename,"%s/nlist_dist.dat",dirname);
    fp9 = fopen(filename,"w");
    sprintf(filename,"%s/result_param.dat",dirname);
    fp10 = fopen(filename,"w");
    sprintf(filename,"%s/rp.dat",dirname);
    fp11 = fopen(filename,"w");
    sprintf(filename,"%s/rpO.dat",dirname);
    fp12 = fopen(filename,"w");

    for (i=0;i<ncluster2;i++) {
        fprintf(fp1,"%g %g %g %g\n",*(cluster2+4*i+0),*(cluster2+4*i+1),*(cluster2+4*i+2),*(cluster2+4*i+3));
        fprintf(fp3,"%d %d\n",*(store_pos2+2*i+0),*(store_pos2+2*i+1));
    }
    for (i=0;i<ncluster3;i++) {
        fprintf(fp2,"%g %g %g\n",*(cluster3+3*i+0),*(cluster3+3*i+1),*(cluster3+3*i+2));
        fprintf(fp4,"%d %d %d\n",*(store_pos3+3*i+0),*(store_pos3+3*i+1),*(store_pos3+3*i+2));
    }
    for (i=0;i<np;i++) {
        for (j=0;j<neighbor_num;j++)
            fprintf(fp5,"%d ",map_to_cluster2[i][j]);
        fprintf(fp5,"\n");
    }
    for (k=0;k<neighbor_num;k++)
        for (i=0;i<np;i++) {
            for (j=0;j<neighbor_num;j++)
                fprintf(fp6,"%d ",*(map_to_cluster3+np*neighbor_num*k+neighbor_num*i+j));
            fprintf(fp6,"\n");
        }
    for (i=0;i<np;i++)
        fprintf(fp7,"%d\n",map_to_cluster1[i]);
    for (i=0;i<np;i++) {
        for (j=0;j<neighbor_num;j++)
            fprintf(fp8,"%d ",nlist[i][j]);
        fprintf(fp8,"\n");
    }
    for (i=0;i<np;i++) {
        for (j=0;j<neighbor_num;j++)
            fprintf(fp9,"%f ",nlist_dist[i][j]);
        fprintf(fp9,"\n");
    }

    fprintf(fp10,"np\t\t%d\n",np);
    fprintf(fp10,"neighbor_num\t\t%d\n",neighbor_num);
    fprintf(fp10,"ncluster1\t\t%d\n",ncluster1);
    fprintf(fp10,"ncluster2\t\t%d\n",ncluster2);
    fprintf(fp10,"ncluster3\t\t%d\n",ncluster3);
    
    for (i=0;i<np;i++) {
        fprintf(fp11,"%.4f %.4f %.4f\n",*(rp1+i),*(rp2+i),*(rp3+i));
        fprintf(fp12,"%.4f %.4f %.4f\n",*(rpO1+i),*(rpO2+i),*(rpO3+i));
    }
}

void load_double_mat(double *A, int Arow, int Acol, int Apgs, char *datfilename) {
    
    FILE *fp;
    int i, j, k;
    double tmp;
    
    printf("loading filename: %s\n", datfilename);
    fp = fopen(datfilename, "r");
    for (k=0;k<Apgs;k++)
        for (i=0;i<Arow;i++)
            for (j=0;j<Acol;j++) {
                fscanf(fp, "%lf", &tmp);
                *(A+Arow*Acol*k+Acol*i+j) = tmp;
//                printf("load data [%d][%d][%d] = %f\n",i,j,k,tmp);
            }
    
    fclose(fp);
    
}






        
        
        
        
        
        
        
        
        
        
        
        
        
        
