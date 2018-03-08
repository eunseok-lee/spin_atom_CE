/*
 findcluster_ternary_par.c
 written by Eunseok Lee
 
 function: find the most representative cluster function
 
 v1: Feb 2, 2018
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
//#include <cem.h>
#include <mpi.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

void randperm(int*, int);
void load_double_mat(double*, int, int, int, char*);
void load_int_mat(int*, int, int, int, char*);
void load_double_array(double*, int, char*);
void load_int_array(int*, int, char*);
void load_double_value(int,char);
void load_int_value(int,char);
int mat_size_row(int*);
int find_value_in_array(int*,int,int,int);
void mat2d_prod(double*, int, int, double*, int, int, double*, int, int);
double obtainerr2_par(int,int,double*, int, int, int*, int, double*);
void least_square_solver(double*, int, int, double*, double*);

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

int main(int argc, char **argv)
{
    int n1, n2, n3;
    int np, nfu;
    double kv[3][3], pk[3][3], pp[8][3];
    double *rp, *rpL, *rpN, *rpO;
    int *map_to_cluster1, *map_to_cluster2, *map_to_cluster3;
    int *nlist;
    int neighbor_num;
    int ncluster, ncluster2, ncluster3, ncorr_col;
    int c2start, c3start;
    int data;
    double *E, *Ef;
    int ndata;
    int i, j, k, nj, nk, ctn;
    int iter, max_iter;
    double kT=0.0256;   // Default value of kT
    int howmanycluster;
    int dispfreq;
    char buff_line[200], dummy[200], param_name[100];
    int param_name_len;
    double cvs_tol = 0.01;  // Default value of cvs_tol
    int construct_corr_mat = 0; //Default value of construct_corr_mat

    // mpi parameters and initialization
    int numprocs, rank, mtype;
    int row_dist_size;
    int row_ini, row_end, row_offset;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);  // Get # processors
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);      // Get my rank (id)

    if (numprocs>1)
        printf("Parallel computing: the input files are read by all nodes. Multiple messages are not due to error!\n",numprocs);

    // Construct conrrelation matrix. It can also be loaded.
    
    int n_corr_mat_ug_row;
    int n_non_singular_col;
    FILE *fp;
    char dirname[100]="dir_inputs";
    char paramfilename[100];
    sprintf(paramfilename,"%s/param.dat",dirname);
    char corr_mat_ugs_filename[200];
    fp = fopen(paramfilename, "r");
    while(fgets(buff_line,sizeof(buff_line),fp) != NULL) {
        if (buff_line[0] == '#') {
            //            printf("Comment line: %s",buff_line);
            continue;
        }
        strcpy(param_name,"corr_mat_ugs_filename");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %s",dummy,corr_mat_ugs_filename);
            if (rank == MASTER)
                printf("corr_mat_ugs_filename = %s\n",corr_mat_ugs_filename);
        }
        strcpy(param_name,"n_corr_mat_ug_row");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&n_corr_mat_ug_row);
            if (rank == MASTER)
                printf("n_corr_mat_ug_row = %d\n",n_corr_mat_ug_row);
        }
        strcpy(param_name,"n_non_singular_col");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&n_non_singular_col);
            if (rank == MASTER)
                printf("n_non_singular_col = %d\n",n_non_singular_col);
        }
        strcpy(param_name,"howmanycluster");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&howmanycluster);
            if (rank == MASTER)
                printf("howmanycluster = %d\n",howmanycluster);
        }
        strcpy(param_name,"max_iter");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&max_iter);
            if (rank == MASTER)
                printf("max_iter = %d\n",max_iter);
        }
        strcpy(param_name,"kT");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %lf",dummy,&kT);
            if (rank == MASTER)
                printf("kT = %f\n",kT);
        }
        strcpy(param_name,"cvs_tol");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %lf",dummy,&cvs_tol);
            if (rank == MASTER)
                printf("cvs_tol = %f\n",cvs_tol);
        }
        strcpy(param_name,"dispfreq");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&dispfreq);
            if (rank == MASTER)
                printf("dispfreq = %d\n",dispfreq);
        }
    }
    fclose(fp);

    char loadfilename[100];
    double *corr_mat_ugs;
    corr_mat_ugs = (double*) malloc(n_corr_mat_ug_row*n_non_singular_col*sizeof(double));
    sprintf(loadfilename,"%s/%s",dirname,corr_mat_ugs_filename);
    load_double_mat(corr_mat_ugs,n_corr_mat_ug_row,n_non_singular_col,1,loadfilename);
    
    int usefulcorr_col[n_non_singular_col];
    sprintf(loadfilename,"%s/usefulcorr_col.dat",dirname);
    load_int_array(usefulcorr_col,n_non_singular_col,loadfilename);
    
    double Ef_ug[n_corr_mat_ug_row];
    sprintf(loadfilename,"%s/Ef_ug.dat",dirname);
    load_double_array(Ef_ug,n_corr_mat_ug_row,loadfilename);
    
/*    int nC_ug[n_corr_mat_ug_row];
    sprintf(loadfilename,"%s/nC_ug.dat",dirname);
    load_int_array(nC_ug,n_corr_mat_ug_row,loadfilename);
*/
    row_offset = (int) ceil(n_corr_mat_ug_row*1.0/numprocs);
    if (numprocs>1 && rank==MASTER)
        printf("Correlation Matrix is calculated over %d processors: row_offset = %d\n",numprocs,row_offset);
    double err2_sum_from_master;
    double err2_sum_from_worker;
    int cluster_set1[howmanycluster];
    
    /**** Start of MC simulation ****/
    if (rank == MASTER) {
        int cluster_set1_old[howmanycluster], cluster_set1_min[howmanycluster], cluster_set2[n_non_singular_col-howmanycluster];
        double cvs, cvs_old, cvs_min;
        int select1, select2, target, candidate;
        
        for (i=0;i<howmanycluster;i++)
            cluster_set1[i] = i;
        for (i=0;i<n_non_singular_col-howmanycluster;i++)
            cluster_set2[i] = i+howmanycluster;
        
        mtype = FROM_MASTER;
        for (i=1;i<numprocs;i++) {
            MPI_Send(&cluster_set1, howmanycluster, MPI_INT, i, mtype, MPI_COMM_WORLD);
        }
        row_ini = rank * row_offset;
        row_end = row_ini +row_offset;
        err2_sum_from_master = obtainerr2_par(row_ini,row_end,corr_mat_ugs,n_corr_mat_ug_row,n_non_singular_col,cluster_set1,howmanycluster,Ef_ug);
//        printf("initial err2 = %f\n",err2_sum_from_master);
        for (i=1;i<numprocs;i++) {
            mtype = FROM_WORKER;
            MPI_Recv(&err2_sum_from_worker, 1, MPI_DOUBLE, i, mtype, MPI_COMM_WORLD, &status);
            err2_sum_from_master += err2_sum_from_worker;
        }
        printf("initial err2 = %f\n",err2_sum_from_master);
        cvs = sqrt(err2_sum_from_master/n_corr_mat_ug_row);
        cvs_min = cvs;
        for (i=0;i<howmanycluster;i++)
            cluster_set1_min[i] = cluster_set1[i];
        printf("initial cvs = %f\n",cvs);
        for (iter=0;iter<max_iter;iter++) {
            select1 = (int) floor(drand48()*howmanycluster);
            select2 = (int) floor(drand48()*(n_non_singular_col-howmanycluster));
            target = cluster_set1[select1];
            candidate = cluster_set2[select2];
            cvs_old = cvs;
            cluster_set1[select1] = candidate;
            mtype = FROM_MASTER;
            for (i=1;i<numprocs;i++) {
                MPI_Send(&cluster_set1, howmanycluster, MPI_INT, i, mtype, MPI_COMM_WORLD);
            }
            row_ini = rank * row_offset;
            row_end = row_ini +row_offset;
            err2_sum_from_master = obtainerr2_par(row_ini,row_end,corr_mat_ugs,n_corr_mat_ug_row,n_non_singular_col,cluster_set1,howmanycluster,Ef_ug);
            for (i=1;i<numprocs;i++) {
                mtype = FROM_WORKER;
                MPI_Recv(&err2_sum_from_worker, 1, MPI_DOUBLE, i, mtype, MPI_COMM_WORLD, &status);
                err2_sum_from_master += err2_sum_from_worker;
            }
            cvs = sqrt(err2_sum_from_master/n_corr_mat_ug_row);
            if (cvs < cvs_old)
                cluster_set2[select2] = target;
            else if (exp(-(cvs-cvs_old)/kT) > drand48())
                cluster_set2[select2] = target;
            else {
                cluster_set1[select1] = target;
                cvs = cvs_old;
            }
            if (cvs < cvs_min) {
                for (i=0;i<howmanycluster;i++) 
                    cluster_set1_min[i] = cluster_set1[i];
                cvs_min = cvs;
            }
            if (cvs < cvs_tol)
                break;
            if (iter%dispfreq==0)
                printf("iter=%8d, cvs=%f, cvs_min=%f, cvs_min/fu=%f\n",iter,cvs,cvs_min,cvs_min/72);
        }
        
        // print the result
        printf("End of MC iteration. The selected clusters are...\n");
        printf("ID in non-singular | ID in entire\n");
        for (i=0;i<howmanycluster;i++)
            printf("    %4d                %4d\n",cluster_set1_min[i],usefulcorr_col[cluster_set1_min[i]]);
        
        double err_pred, rms;
        double Ef_pred[n_corr_mat_ug_row];
        double *corr_mat_ugs_reduced;
        corr_mat_ugs_reduced = (double*) malloc(n_corr_mat_ug_row*howmanycluster*sizeof(double));
        double eci[howmanycluster];
        for (i=0;i<n_corr_mat_ug_row;i++) {
            for (j=0;j<howmanycluster;j++)
                *(corr_mat_ugs_reduced+howmanycluster*i+j) = *(corr_mat_ugs+n_non_singular_col*i+cluster_set1_min[j]);
        }
        least_square_solver(corr_mat_ugs_reduced,n_corr_mat_ug_row,howmanycluster,Ef_ug,eci);
        mat2d_prod(Ef_pred,n_corr_mat_ug_row,1,corr_mat_ugs_reduced,n_corr_mat_ug_row,howmanycluster,eci,howmanycluster,1);
        err_pred = 0.0;
        for (i=0;i<n_corr_mat_ug_row;i++)
            err_pred += (Ef_ug[i] - Ef_pred[i])*(Ef_ug[i] - Ef_pred[i]);
        rms = sqrt(err_pred/n_corr_mat_ug_row);

        // Write result into files
        FILE *fp2, *fp3;
        char cluster_set_filename[200];
        char eci_filename[100];
        strcpy(dirname,"dir_result");
        struct stat st = {0};
        if (stat(dirname, &st) == -1) {
            mkdir(dirname,0777);
            printf("Created a new directory for storing result.\n");
        }
        sprintf(cluster_set_filename, "%s/cluster_set_N%d.dat",dirname,howmanycluster);
        sprintf(eci_filename,"%s/x.dat",dirname);
        fp2 = fopen(cluster_set_filename,"w");
        fp3 = fopen(eci_filename,"w");
        for (i=0;i<howmanycluster;i++) {
            fprintf(fp2,"%4d\t%4d\n",cluster_set1_min[i],usefulcorr_col[cluster_set1_min[i]]);
            fprintf(fp3,"%.4f\n",eci[i]);
        }
    }
    else {
        for (iter=0;iter<(max_iter+1);iter++) {
            mtype = FROM_MASTER;
            MPI_Recv(&cluster_set1, howmanycluster, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
            row_ini = rank * row_offset;
            row_end = row_ini + row_offset;
            if (rank == numprocs-1)
                row_end = n_corr_mat_ug_row;  // last processor till the last row, -1 is taken inside obtainerr2_par
            err2_sum_from_worker = obtainerr2_par(row_ini,row_end,corr_mat_ugs,n_corr_mat_ug_row,n_non_singular_col,cluster_set1,howmanycluster,Ef_ug);
            mtype = FROM_WORKER;
            MPI_Send(&err2_sum_from_worker, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
    
}

void load_double_mat(double *A, int Arow, int Acol, int Apgs, char *datfilename) {
    
    FILE *fp;
    int i, j, k;
    double tmp;
    
    printf("filename: %s\n", datfilename);
    fp = fopen(datfilename, "r");
    for (k=0;k<Apgs;k++)
        for (i=0;i<Arow;i++)
            for (j=0;j<Acol;j++) {
                fscanf(fp, "%lf", &tmp);
                *(A+Arow*Acol*k+Acol*i+j) = tmp;
            }
    
    fclose(fp);
    
}

void load_int_mat(int *A, int Arow, int Acol, int Apgs, char *datfilename) {
    
    FILE *fp;
    int i, j, k, tmp;
    
    printf("filename: %s\n", datfilename);
    fp = fopen(datfilename, "r");
    for (k=0;k<Apgs;k++)
        for (i=0;i<Arow;i++)
            for (j=0;j<Acol;j++) {
                fscanf(fp, "%d", &tmp);
                *(A+Arow*Acol*k+Acol*i+j) = tmp;
            }
    
    fclose(fp);
    
}

void load_int_array(int *array, int arraysize, char *datfilename) {
    
    FILE *fp;
    int i;
    
    printf("filename: %s\n", datfilename);
    fp = fopen(datfilename, "r");
    //    printf("step2\n");
    for (i=0;i<arraysize;i++) {
        fscanf(fp, "%d", &array[i]);
        //        printf("%d\n",array[i]);
    }
    
    fclose(fp);
    //    printf("step3\n");
}

void load_double_array(double *array, int arraysize, char *datfilename) {
    
    FILE *fp;
    int i;
    
    printf("filename: %s\n", datfilename);
    fp = fopen(datfilename, "r");
    //    printf("step4\n");
    for (i=0;i<arraysize;i++) {
        fscanf(fp, "%lf", &array[i]);
        //        printf("%f\n",array[i]);
    }
    
    fclose(fp);
    //    printf("step5\n");
}

void randperm(int *a, int np) {
    
    int i, id1, id2, id3, tmpr, num_perm;
    
    num_perm = 100*np;
    
    for (i=0;i<np;i++)
        *(a+i) = i;
    
    for (i=0;i<num_perm;i++) {
        id1 = rand() % np;
        id3 = rand() % np;
        id2 = (id1 + id3) % np;
        if (id1 != id2) {
            tmpr = *(a+id2);
            *(a+id2) = *(a+id1);
            *(a+id1) = tmpr;
        }
    }
}

int find_value_in_array(int *A, int target_rowA, int colA, int target_value) {
    int i;
    
    for (i=0;i<colA;i++)
        if (*(A+colA*target_rowA+i)==target_value)
            return(i);
    if (i==colA)
        return(-1);
}


























