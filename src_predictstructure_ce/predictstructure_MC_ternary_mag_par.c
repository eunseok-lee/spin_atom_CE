/*
 predictstructure_MC_ternary_mag_par.c
 by Eunseok Lee
 
 function: predict the lowest energy structure using given CE model
 
 v1: Feb 2, 2018
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include "mpi.h"
//#include <cem.h>

void randperm(int*, int);
void obtain_corr_mat_mag_par(int, int, double*, int, int, int*, int*, int, int, int*, int*, int*, int, int);
void mat_copy_double(double*, double*, int, int, int);
void mat_copy_int(int*, int*, int, int, int);
void mat2d_prod(double*,int,int,double*,int,int,double*,int,int);
void sort_array(double*, int);
void sort_int_array(int*, int);
void load_int_array(int*, int, char*);
void load_double_array(double*, int, char*);
void load_int_mat(int*, int, int, int, char*);
void update_corr_mat(double*, int*, int, int);
double mat2d_sum_row(double*, int, int);

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

int main(int argc, char **argv) {
    
    int i, j, k, l, m, n, nj, nk, tmp;
    int iter, max_iter, dispfreq, newstart;
    double kT;
    int np, howmanyLi, howmanyCo, howmanyNi;
    int howmanycluster, howmanyclustercol, ncorr_col, neighbor_num, c2start, c3start;
    int ncluster1, ncluster2, ncluster3;
    int corr_cal_algo = 0;
    int ctr;
    int num_ni4, max_num_ni4;
    char buff_line[200], dummy[200], param_name[100];
    int param_name_len;

    // mpi parameters and initialization
    int numprocs, rank, mtype;
    int row_dist_size;
    int row_ini, row_end, row_offset;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);  // Get # processors
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);      // Get my rank (id)
    
    // MC parameters
    FILE *fp;
    char dirname[100]="dir_inputs";
    char paramfilename[100];
    char data_ini_filename[100];
    char magmom_ini_filename[100];
    sprintf(paramfilename,"%s/param.dat",dirname);
    fp = fopen(paramfilename, "r");
    while(fgets(buff_line,sizeof(buff_line),fp) != NULL) {
        if (buff_line[0] == '#') {
            //            printf("Comment line: %s",buff_line);
            continue;
        }
        strcpy(param_name,"howmanyCo");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&howmanyCo);
            if (rank == MASTER)
                printf("howmanyCo = %d\n",howmanyCo);
        }
        strcpy(param_name,"newstart");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&newstart);
            if (rank == MASTER)
                printf("newstart = %d\n",newstart);
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
        strcpy(param_name,"dispfreq");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&dispfreq);
            if (rank == MASTER)
                printf("dispfreq = %d\n",dispfreq);
        }
        strcpy(param_name,"np");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&np);
            if (rank == MASTER)
                printf("np = %d\n",np);
        }
/*        strcpy(param_name,"ncorr_col");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&ncorr_col);
            if (rank == MASTER)
                printf("ncorr_col = %d\n",ncorr_col);
        }*/
        strcpy(param_name,"neighbor_num");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&neighbor_num);
            if (rank == MASTER)
                printf("neighbor_num = %d\n",neighbor_num);
        }
        strcpy(param_name,"howmanyclustercol");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&howmanyclustercol);
            if (rank == MASTER)
                printf("howmanyclustercol = %d\n",howmanyclustercol);
        }
        strcpy(param_name,"ncluster1");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&ncluster1);
            if (rank == MASTER)
                printf("ncluster1 = %d\n",ncluster1);
        }
        strcpy(param_name,"ncluster2");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&ncluster2);
            if (rank == MASTER)
                printf("ncluster2 = %d\n",ncluster2);
        }
        strcpy(param_name,"ncluster3");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %d",dummy,&ncluster3);
            if (rank == MASTER)
                printf("ncluster3 = %d\n",ncluster3);
        }
        strcpy(param_name,"data_ini_filename");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %s",dummy,data_ini_filename);
            if (rank == MASTER)
                printf("data_ini_filename = %s\n",data_ini_filename);
        }
        strcpy(param_name,"magmom_ini_filename");
        param_name_len = strlen(param_name);
        strncpy(dummy,buff_line,param_name_len);
        dummy[param_name_len] = '\0';
        if (strcmp(dummy,param_name)==0) {
            sscanf(buff_line,"%s %s",dummy,magmom_ini_filename);
            if (rank == MASTER)
                printf("magmom_ini_filename = %s\n",magmom_ini_filename);
        }
    }
    fclose(fp);

    howmanyLi = np/2;
    howmanyNi = howmanyLi - howmanyCo;
    ncorr_col = 1 + 4*ncluster1 + 9*ncluster2 + 16*ncluster3;
    printf("ncorr_col=%d, based from ncluster1~3\n",ncorr_col);
    c2start = 4*ncluster1 + 1;      // Don't need -1: cluster2 idx starts from 0
    c3start = 9*ncluster2 + c2start;

    if (rank == MASTER) {
        printf("howmanyCo=%d, howmanyNi=%d, newstart=%d, kT=%f, max_iter=%d, dispfreq=%d, howmanyLi=%d, howmanyNi=%d\n",howmanyCo, howmanyNi, newstart, kT, max_iter, dispfreq, howmanyLi, howmanyNi);
        if (numprocs>1)
            printf("Parallel computing: the task was equally distributed to %d processors.\n",numprocs);
    }

    int data_trial[np], data_trial_min[np], data_trial_old[np];
    int magmom_trial[np], magmom_trial_min[np], magmom_trial_old[np];
    int rand_dist[np];
    double corr_mat_trial[ncorr_col], corr_mat_trial_r[howmanyclustercol], corr_mat_trial_min[ncorr_col], corr_mat_trial_old[ncorr_col];
    int *nlist;
    int *map_to_cluster1, *map_to_cluster2, *map_to_cluster3;
    nlist           = (int*) malloc(np*neighbor_num*sizeof(int));
    map_to_cluster1 = (int*) malloc(np*sizeof(int));
    map_to_cluster2 = (int*) malloc(np*neighbor_num*sizeof(int));
    map_to_cluster3 = (int*) malloc(np*neighbor_num*neighbor_num*sizeof(int));
//    int cluster_set1_min[howmanycluster];
    int cluster_set1_min_col[howmanyclustercol];
    double x[howmanyclustercol];
    double Ef_trial, Ef_trial_old, Ef_trial_min;
    
    // load the result from findcluster_*.c
//    load_int_array(cluster_set1_min, howmanycluster, "cluster_set1_min.dat");
    if (numprocs>1)
        printf("Parallel computing: the input files are read by all nodes. Multiple messages are not due to error!\n");
    char loadfilename[100];
    sprintf(loadfilename,"%s/cluster_set_min.dat",dirname);
    load_int_array(cluster_set1_min_col, howmanyclustercol, loadfilename);
    sprintf(loadfilename,"%s/x.dat",dirname);
    load_double_array(x, howmanyclustercol, loadfilename);
    sprintf(loadfilename,"%s/nlist.dat",dirname);
    load_int_mat(nlist, np, neighbor_num, 1, loadfilename);
    sprintf(loadfilename,"%s/map_to_cluster1.dat",dirname);
    load_int_array(map_to_cluster1, np, loadfilename);
    sprintf(loadfilename,"%s/map_to_cluster2.dat",dirname);
    load_int_mat(map_to_cluster2, np, neighbor_num, 1, loadfilename);
    sprintf(loadfilename,"%s/map_to_cluster3.dat",dirname);
    load_int_mat(map_to_cluster3, np, neighbor_num, neighbor_num, loadfilename);
    
    if (rank == MASTER && newstart == 1) {
        randperm(rand_dist,np);
/*        FILE *fptmp;
        fptmp = fopen("rand_dist.dat", "w");
        for (i=0;i<np;i++) {
            fprintf(fptmp,"%d\n",rand_dist[i]);
//            printf("%d\n",rand_dist[i]);
        }
        fclose(fptmp);
 */
        for (i=0;i<howmanyLi;i++) {
            data_trial[rand_dist[i]] = 0;
            magmom_trial[rand_dist[i]] = 0;
        }
        for (i=howmanyLi;i<howmanyLi+howmanyCo;i++) {
            data_trial[rand_dist[i]] = 1;
            magmom_trial[rand_dist[i]] = 0;
        }
        for (i=howmanyLi+howmanyCo;i<np;i++) {
            data_trial[rand_dist[i]] = -1;
            magmom_trial[rand_dist[i]] = 1;
        }
    }
    else {
        sprintf(loadfilename,"%s/%s",dirname,data_ini_filename);
        load_int_array(data_trial, np, loadfilename);
        sprintf(loadfilename,"%s/%s",dirname,magmom_ini_filename);
        load_int_array(magmom_trial, np, loadfilename);
    }

    printf("data_trial and magmom_trial were initialized.\n");
    
    // obtain the correlation matrix through parallel computing
    row_offset = (int) ceil(np*1.0/numprocs);
    double tmp_corr_mat_trial[numprocs][ncorr_col];
    double tmp_corr_mat_trial_task[ncorr_col];
    
    if (rank == MASTER) {
        strcpy(dirname,"dir_result");
        struct stat st = {0};
        if (stat(dirname, &st) == -1) {
            mkdir(dirname,0777);
            printf("Created a new directory for storing result.\n");
        }
		char on_the_fly_filename[100];
        // nullify temporary corr_mat_trial per each core
        for (i=0;i<numprocs;i++)
            for (j=0;j<ncorr_col;j++)
                tmp_corr_mat_trial[i][j] = 0.0;
        mtype = FROM_MASTER;
        for (i=1;i<numprocs;i++) {
            MPI_Send(&data_trial, np, MPI_INT, i, mtype, MPI_COMM_WORLD);
            MPI_Send(&magmom_trial, np, MPI_INT, i, mtype, MPI_COMM_WORLD);
        }
        row_ini = rank * row_offset;
        row_end = row_ini +row_offset;
//        printf("MASTER: initialized tmp_corr_mat_trial, task rows:%d-%d\n",row_ini,row_end);
        obtain_corr_mat_mag_par(row_ini,row_end,corr_mat_trial,1,ncorr_col,data_trial,magmom_trial,c2start,c3start,nlist,map_to_cluster2,map_to_cluster3,np,neighbor_num);
//        printf("MASTER: finished the task\n");
        for (i=1;i<numprocs;i++) {
            mtype = FROM_WORKER;
            MPI_Recv(&tmp_corr_mat_trial[i][0], ncorr_col, MPI_DOUBLE, i, mtype, MPI_COMM_WORLD, &status);
        }
//        printf("MASTER: Received the task from WORKERs\n");
//        for (j=0;j<ncorr_col;j++)
//            corr_mat_trial[j] = tmp_corr_mat_trial_task[j];

        for (j=1;j<ncorr_col;j++)   //corr_mat for empty cluster from workers must be excluded
            for (i=1;i<numprocs;i++)
                corr_mat_trial[j] += tmp_corr_mat_trial[i][j];
        printf("Initial corr_mat was created. \n");
        char corr_mat0_filename[100];
        sprintf(corr_mat0_filename,"%s/corr_mat_trial0.dat",dirname);
		fp=fopen(corr_mat0_filename,"w");
		for (i=0;i<ncorr_col;i++)
			fprintf(fp,"%f\n",corr_mat_trial[i]);
		fclose(fp);
        for (i=0;i<howmanyclustercol;i++)
            corr_mat_trial_r[i] = corr_mat_trial[cluster_set1_min_col[i]];
//        printf("Processed step 0. \n");
        //    for (i=0;i<howmanyclustercol;i++) {
        //        printf("corr_mat_trial_r[%d] = %g \n",i,corr_mat_trial_r[i]);
        //    }

        Ef_trial = 0.0;
        for (i=0;i<howmanyclustercol;i++)
            Ef_trial += corr_mat_trial_r[i]*x[i];
        printf("Ef_trial initial = %g\n",Ef_trial);
        //    mat2d_prod(Ef_trial,1,1,corr_mat_trial_r,1,howmanyclustercol,x,howmanyclustercol,1);
//        printf("Processed step 1. \n");
        mat_copy_int(data_trial_min,data_trial,1,np,1);
        mat_copy_int(magmom_trial_min,magmom_trial,1,np,1);
        mat_copy_double(corr_mat_trial_min,corr_mat_trial,1,ncorr_col,1);
        Ef_trial_min = Ef_trial;

        int n_event_accept = 0;
        int swap_case0, swap_case1, swap_case_set[6][2], n_swap_case_set[6][2];
        int swap_i, n_swap_is, swap_i_id, swap_j, n_swap_js, swap_j_id;
        int casenum;
        double event_type;
        
        swap_case_set[0][0] = -1; swap_case_set[0][1] =  0;
        swap_case_set[1][0] = -1; swap_case_set[1][1] =  1;
        swap_case_set[2][0] =  0; swap_case_set[2][1] =  1;
        swap_case_set[3][0] =  0; swap_case_set[3][1] = -1;
        swap_case_set[4][0] =  1; swap_case_set[4][1] = -1;
        swap_case_set[5][0] =  1; swap_case_set[5][1] =  0;
/*
        n_swap_case_set[0][0] = howmanyNi; n_swap_case_set[0][1] = howmanyLi;
        n_swap_case_set[1][0] = howmanyNi; n_swap_case_set[1][1] = howmanyCo;
        n_swap_case_set[2][0] = howmanyLi; n_swap_case_set[2][1] = howmanyCo;
        n_swap_case_set[3][0] = howmanyLi; n_swap_case_set[3][1] = howmanyNi;
        n_swap_case_set[4][0] = howmanyCo; n_swap_case_set[4][1] = howmanyNi;
        n_swap_case_set[5][0] = howmanyCo; n_swap_case_set[5][1] = howmanyLi;
*/
        num_ni4 = 0;
        for (i=0;i<np;i++)
            if (data_trial[i]==-1 && magmom_trial[i]==0)
                num_ni4++;
        max_num_ni4 = (int) floor(howmanyNi/2.0);
        
        for (iter=0;iter<max_iter;iter++) {
            if (iter%dispfreq==0)
                printf("MC step: %d, Ef_trial = %f, Ef_trial_min = %f\n",iter,Ef_trial,Ef_trial_min);
            mat_copy_int(data_trial_old, data_trial, 1, np, 1);
            mat_copy_int(magmom_trial_old, magmom_trial, 1, np, 1);
            mat_copy_double(corr_mat_trial_old, corr_mat_trial, 1, ncorr_col, 1);
            Ef_trial_old = Ef_trial;
            ctr = 0;
    //        printf("Creating an event ~ ");
            while (ctr == 0) {
                event_type = (double) rand()/RAND_MAX;
	            if (event_type < 0.25) {
    	            if (howmanyCo == np/2) {
        	            swap_case0 = 0; swap_case1 = 1;
            	        n_swap_is = np/2; n_swap_js = np/2;
                	}
	                else if (howmanyCo == 0) {
    	                swap_case0 = 0; swap_case1 = -1;
        	            n_swap_is = np/2; n_swap_js = np/2;
            	    }
                	else {
	                    i = rand() % 6;
    	                swap_case0 = swap_case_set[i][0];
        	            swap_case1 = swap_case_set[i][1];
            	        n_swap_is = 0; n_swap_js = 0;
                	    for (i=0;i<np;i++) {
                    	    if (data_trial[i]==swap_case0)
                        	    n_swap_is++;
	                        if (data_trial[i]==swap_case1)
    	                        n_swap_js++;
        	            }
            	    }
                	swap_i_id = rand() % n_swap_is;
	                swap_j_id = rand() % n_swap_js;
    	            
        	        l = 0;
            	    for (i=0;i<np;i++)
                	    if (data_trial[i] == swap_case0) {
                    	    if (l == swap_i_id) {
                        	    swap_i = i;
                            	break;
	                        }
    	                    else
        	                    l++;
            	        }
                	l = 0;
	                for (j=0;j<np;j++)
    	                if (data_trial[j] == swap_case1) {
        	                if (l == swap_j_id) {
            	                swap_j = j;
                	            break;
                    	    }
                        	else
                            	l++;
	                    }
                    ctr = 1;
        	    }
            	else if (event_type < 0.5) {
                	n_swap_is = 0; n_swap_js = 0;
	                while (n_swap_is == 0 || n_swap_js == 0) {
    	                i = rand() % 6;
        	            swap_case0 = swap_case_set[i][0];
            	        swap_case1 = swap_case_set[i][1];
                	    n_swap_is = 0; n_swap_js = 0;
                    	for (i=0;i<np;i++) {
                        	if (magmom_trial[i]==swap_case0)
	                            n_swap_is++;
    	                    if (magmom_trial[i]==swap_case1)
        	                    n_swap_js++;
            	        }
                	}
	                swap_i_id = rand() % n_swap_is;
    	            swap_j_id = rand() % n_swap_js;
        	            
            	    l = 0;
                	for (i=0;i<np;i++)
                    	if (magmom_trial[i] == swap_case0) {
                        	if (l == swap_i_id) {
                            	swap_i = i;
	                            break;
    	                    }
        	                else
            	                l++;
                	    }
	                l = 0;
    	            for (j=0;j<np;j++)
        	            if (magmom_trial[j] == swap_case1) {
            	            if (l == swap_j_id) {
                	            swap_j = j;
                    	        break;
                        	}
	                        else
    	                        l++;
        	            }
                    ctr = 2;
            	}	
	            else if (event_type < 0.75) {
    	            n_swap_is = 0;
        	        while (n_swap_is==0) {
            	        i = rand() % 6;
                	    swap_case0 = swap_case_set[i][0];
                    	swap_case1 = swap_case_set[i][1];
	                    n_swap_is = 0;
    	                for (i=0;i<np;i++) {
        	                if (magmom_trial[i]==swap_case0)
            	                n_swap_is++;
                	    }
	                }
    	            swap_i_id = rand() % n_swap_is;
        	        
            	    l = 0;
                	for (i=0;i<np;i++)
                    	if (magmom_trial[i] == swap_case0) {
                        	if (l == swap_i_id) {
                            	swap_i = i;
	                            break;
    	                    }
        	                else
            	                l++;
                	    }
                    ctr = 3;
	            }
    	        else {
        	        if (howmanyCo == np/2) {
            	        swap_case0 = 0; swap_case1 = 1;
                	    n_swap_is = np/2; n_swap_js = np/2;
	                }
    	            else if (howmanyCo == 0) {
        	            swap_case0 = 0; swap_case1 = -1;
            	        n_swap_is = np/2; n_swap_js = np/2;
                	}
	                else {
    	                i = rand() % 6;
        	            swap_case0 = swap_case_set[i][0];
            	        swap_case1 = swap_case_set[i][1];
                	    n_swap_is = 0; n_swap_js = 0;
                    	for (i=0;i<np;i++) {
                        	if (data_trial[i]==swap_case0)
                            	n_swap_is++;
	                        if (data_trial[i]==swap_case1)
    	                        n_swap_js++;
        	            }
            	    }
                	swap_i_id = rand() % n_swap_is;
	                swap_j_id = rand() % n_swap_js;
    	            
        	        l = 0;
            	    for (i=0;i<np;i++)
                	    if (data_trial[i] == swap_case0) {
                    	    if (l == swap_i_id) {
                        	    swap_i = i;
                            	break;
	                        }
    	                    else
        	                    l++;
            	        }
                	l = 0;
	                for (j=0;j<np;j++)
    	                if (data_trial[j] == swap_case1) {
        	                if (l == swap_j_id) {
            	                swap_j = j;
                	            break;
                    	    }
                        	else
                            	l++;
	                    }
    	            ctr = 4;    // The spins will be swapped as well. So the spin compatibility check is not needed.
//                printf(" type: 4");
        	    }

                if (corr_cal_algo == 0) {
                    switch(ctr) {
                        case 1:
                            data_trial[swap_i] = swap_case1;
                            data_trial[swap_j] = swap_case0;
                            break;
                        case 2:
                            magmom_trial[swap_i] = swap_case1;
                            magmom_trial[swap_j] = swap_case0;
                            break;
                        case 3:
                            magmom_trial[swap_i] = swap_case1;
                            break;
                        case 4:
                            data_trial[swap_i] = swap_case1;
                            data_trial[swap_j] = swap_case0;
                            tmp = magmom_trial[swap_i];
                            magmom_trial[swap_i] = magmom_trial[swap_j];
                            magmom_trial[swap_j] = tmp;
                            break;
                    }
                }
                else if (corr_cal_algo == 1) {
                    /*            switch (ctr) {
                     case 1:
                     data_trial[swap_i] = swap_case1;
                     data_trial[swap_j] = swap_case0;
                     corr_update_atom_swap;
                     break;
                     case 2:
                     magmom_trial[swap_i] = swap_case1;
                     magmom_trial[swap_j] = swap_case0;
                     corr_update_spin_swap;
                     break;
                     case 3:
                     magmom_trial[swap_i] = swap_case1;
                     corr_update_spin_switch;
                     break;
                     case 4:
                     data_trial[swap_i] = swap_case1;
                     data_trial[swap_j] = swap_case0;
                     corr_update_atom_swap;
                     magmom_trial[swap_i] = spin_j;
                     magmom_trial[swap_j] = spin_i;
                     corr_update_spin_swap;
                     break;
                     } */
                }
                // Check if the event is compatible with constraints on configuration
                num_ni4 = 0;
                for (i=0;i<np;i++) {
/*                    if (*(data_trial+i)==0 && *(magmom_trial+i)!=0) {
                        ctr = 0;
						printf("ctr was zeroed: incompatible with Li env.\n");
                        break;
                    }*/
/*                    if (*(data_trial+i)==1 && *(magmom_trial+i)!=1) {
                        ctr = 0;
                        printf("ctr was zeroed: incompatible with Co env.\n");
                        break;
                    }*/
                    if (*(data_trial+i)==-1 && *(magmom_trial+i)==0)
                        num_ni4++;
                }
                if (num_ni4 > max_num_ni4) {
                    ctr = 0;
					printf("ctr was zeroed: incompatible with Ni-antisite max.\n");
				}
                
                if (ctr==0) {
                    mat_copy_int(data_trial, data_trial_old, 1, np, 1);
                    mat_copy_int(magmom_trial, magmom_trial_old, 1, np, 1);
                }

            }
        
            // MPI to obtain corr_mat with updated data and magmom
            for (i=0;i<numprocs;i++)
                for (j=0;j<ncorr_col;j++)
                    tmp_corr_mat_trial[i][j] = 0.0;
            mtype = FROM_MASTER;
            for (i=1;i<numprocs;i++) {
                MPI_Send(&data_trial, np, MPI_INT, i, mtype, MPI_COMM_WORLD);
                MPI_Send(&magmom_trial, np, MPI_INT, i, mtype, MPI_COMM_WORLD);
            }
            row_ini = rank * row_offset;
            row_end = row_ini +row_offset;
//                printf("MASTER: iter%d, initialized tmp_corr_mat_trial, task rows:%d-%d\n",iter,row_ini,row_end);
            obtain_corr_mat_mag_par(row_ini,row_end,corr_mat_trial,1,ncorr_col,data_trial,magmom_trial,c2start,c3start,nlist,map_to_cluster2,map_to_cluster3,np,neighbor_num);
//                printf("MASTER: iter%d, finished the task\n",iter);
            for (i=1;i<numprocs;i++) {
                mtype = FROM_WORKER;
                MPI_Recv(&tmp_corr_mat_trial[i][0], ncorr_col, MPI_DOUBLE, i, mtype, MPI_COMM_WORLD, &status);
            }
//                printf("MASTER: iter%d, Received the task from WORKERs\n",iter);
            //        for (j=0;j<ncorr_col;j++)
            //            corr_mat_trial[j] = tmp_corr_mat_trial_task[j];
            
            for (j=1;j<ncorr_col;j++)
                for (i=1;i<numprocs;i++)
                    corr_mat_trial[j] += tmp_corr_mat_trial[i][j];
//                printf("Initial corr_mat was created. \n");
            
            for (i=0;i<howmanyclustercol;i++) {
                corr_mat_trial_r[i] = corr_mat_trial[cluster_set1_min_col[i]];
//				printf("corr_mat_trial_r[%d]=%f\n",i,corr_mat_trial_r[i]);
			}
            Ef_trial = 0.0;
            for (i=0;i<howmanyclustercol;i++)
                Ef_trial += corr_mat_trial_r[i]*x[i];
//			printf("new Ef_trial = %f, old Ef_trial = %f\n",Ef_trial,Ef_trial_old);
    //        mat2d_prod(Ef_trial,1,1,corr_mat_trial_r,1,howmanyclustercol,x,howmanyclustercol,1);
            
            if (Ef_trial < Ef_trial_old) {
                casenum = 1;
//				printf("event process type%d, ctr type:%d, Ef_trial = %f, Ef_trial_old = %f\n",casenum, ctr, Ef_trial, Ef_trial_old);
                if (Ef_trial < Ef_trial_min) {
                    Ef_trial_min = Ef_trial;
                    mat_copy_int(data_trial_min, data_trial, 1, np, 1);
                    mat_copy_int(magmom_trial_min, data_trial, 1, np, 1);
                    mat_copy_double(corr_mat_trial_min, corr_mat_trial, 1, ncorr_col, 1);
                }
                n_event_accept++;
            }
            else if (exp(-(Ef_trial-Ef_trial_old)/kT) > (double) rand()/RAND_MAX) {
                casenum = 2;
//				printf("event process type%d, ctr type:%d, Ef_trial = %f, Ef_trial_old = %f\n",casenum, ctr, Ef_trial, Ef_trial_old);
                n_event_accept++;
            }
            else {
                casenum = 3;
//				printf("event process type%d, ctr type:%d, Ef_trial = %f, Ef_trial_old = %f\n",casenum, ctr, Ef_trial, Ef_trial_old);
                Ef_trial = Ef_trial_old;
                mat_copy_int(data_trial, data_trial_old, 1, np, 1);
                mat_copy_int(magmom_trial, magmom_trial_old, 1, np, 1);
                mat_copy_double(corr_mat_trial, corr_mat_trial_old, 1, ncorr_col, 1);
            }
/*
			printf("event process type%d: data_trial_old[%d,%d]=[%d,%d], data_trial[%d,%d]=[%d,%d], mag_trial_old[%d,%d]=[%d,%d], mag_trial[%d,%d]=[%d][%d]\n",casenum,swap_i,swap_j,data_trial_old[swap_i],data_trial_old[swap_j],swap_i,swap_j,data_trial[swap_i],data_trial[swap_j],swap_i,swap_j,magmom_trial_old[swap_i],magmom_trial_old[swap_j],swap_i,swap_j,magmom_trial[swap_i],magmom_trial[swap_j]);
			for (i=0;i<howmanyclustercol;i++) {
            	printf("recovered corr_mat_trial_r[%d]=%f\n",i,corr_mat_trial[cluster_set1_min_col[i]]);
        	}
*/
			if (iter%dispfreq==0) {
				sprintf(on_the_fly_filename,"%s/on_the_fly_datamag_%04d.dat",dirname,iter);
				fp = fopen(on_the_fly_filename,"w");
				for (i=0;i<np;i++)
        	    	fprintf(fp,"%d %d\n",data_trial[i],magmom_trial[i]);
        		fclose(fp);
			}	
        }
        
        // display and store the result
        printf("--------------------------------------\n");
        printf("prediction for nC=%d case is done.\n", howmanyCo);
        printf("--------------------------------------\n");

        FILE *fp;
        
        char poscarfilename[100];
        sprintf(poscarfilename,"%s/data_trial_min.dat",dirname);
        fp = fopen(poscarfilename,"w");
        for (i=0;i<np;i++)
            fprintf(fp,"%d\n",data_trial_min[i]);
        fclose(fp);

        char magmomfilename[100];
        sprintf(magmomfilename,"%s/magmom_trial_min.dat",dirname);
        fp = fopen(magmomfilename,"w");
        for (i=0;i<np;i++)
            fprintf(fp,"%d\n",magmom_trial_min[i]);
		fclose(fp);

/*
        char poscar_on_the_fly_filename[100];
        sprintf(poscar_on_the_fly_filename,"data_trial_on_the_fly_Co%dNi%d",howmanyCo,howmanyNi);
        fp = fopen(poscar_on_the_fly_filename,"w");
        for (i=0;i<np;i++)
            fprintf(fp,"%d\n",data_trial[i]);
        fclose(fp);
        
        char magmom_on_the_fly_filename[100];
        sprintf(magmom_on_the_fly_filename,"magmom_trial_on_the_fly_Co%dNi%d",howmanyCo,howmanyNi);
        fp = fopen(magmom_on_the_fly_filename,"w");
        for (i=0;i<np;i++)
            fprintf(fp,"%d\n",magmom_trial[i]);
        fclose(fp);
*/
        
    /*
        sprintf(poscarfilename,"POSCAR_Co%dNi%d",howmanyCo,howmanyNi);
        fp = fopen(filename,"w");
        fprintf(fp,"LiCoxNi1-xO2\n");
        fprintf(fp,"1.0\n");
        fprintf(fp,"%.6f %.6f %.6f\n",kv[0][0],kv[0][1],kv[0][2]);
        fprintf(fp,"%.6f %.6f %.6f\n",kv[1][0],kv[1][1],kv[1][2]);
        fprintf(fp,"%.6f %.6f %.6f\n",kv[2][0],kv[2][1],kv[2][2]);
        if (howmanyCo == 0) {
            fprintf(fp,"Li Ni O\n");
            fprintf(fp,"%d %d %d\n",np/2,np/2,np);
        }
        else if (howmanyCo == np/2) {
            fprintf(fp,"Li Co O\n");
            fprintf(fp,"%d %d %d\n",np/2,np/2,np);
        }
        else {
            fprintf(fp,"Li Co Ni O\n");
            fprintf(fp,"%d %d %d %d\n",np/2,howmanyCo,maxnC-howmanyCo,np);
        }
        fprintf(fp,"Direct\n");
        for (i=0;i<np;i++)
            if (data_trial_min[i] == 0)
                fprintf(fp,"%.6f %.6f %.6f Li\n",rp[i][0],rp[i][1],rp[i][2]);
        if (howmanyCo == np/2)
            for (i=0;i<np;i++)
                if (data_trial_min[i] == 1)
                    fprintf(fp,"%.6f %.6f %.6f Co\n",rp[i][0],rp[i][1],rp[i][2]);
        else if (howmanyCo == 0)
            for (i=0;i<np;i++)
                if (data_trial_min[i] == -1)
                    fprintf(fp,"%.6f %.6f %.6f Ni\n",rp[i][0],rp[i][1],rp[i][2]);
        else {
            for (i=0;i<np;i++)
                if (data_trial_min[i] == 1)
                    fprintf(fp,"%.6f %.6f %.6f Co\n",rp[i][0],rp[i][1],rp[i][2]);
            for (i=0;i<np;i++)
                if (data_trial_min[i] == -1)
                    fprintf(fp,"%.6f %.6f %.6f Ni\n",rp[i][0],rp[i][1],rp[i][2]);
        }
        for (i=0;i<npO;i++)
            fprintf(fp,"%.6f %.6f %.6f O\n",rpO[i][0],rpO[i][1],rpO[i][2]);
    */
        
    }
    else {
        for (iter=0;iter<(max_iter+1);iter++) {
            mtype = FROM_MASTER;
            MPI_Recv(&data_trial, np, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&magmom_trial, np, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
            row_ini = rank * row_offset;
            row_end = row_ini + row_offset;
            if (rank == numprocs-1)
                row_end = np;   // last processor till np, -1 is taken inside obtain_corr_mat_mag_par
            mtype = FROM_MASTER;
//            printf("WORKER%d: Received the Message, task rows:%d-%d\n",rank,row_ini,row_end);
            obtain_corr_mat_mag_par(row_ini,row_end,tmp_corr_mat_trial_task,1,ncorr_col,data_trial,magmom_trial,c2start,c3start,nlist,map_to_cluster2,map_to_cluster3,np,neighbor_num);
//            printf("WORKER%d: finished the task\n",rank);
            mtype = FROM_WORKER;
            MPI_Send(&tmp_corr_mat_trial_task, ncorr_col, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//            printf("WORKER%d: Sent the result\n",rank);
        }
    }
    MPI_Finalize();
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

void mat_copy_double(double *A, double *B, int a, int b, int c) {

    int i;

    for (i=0;i<a*b*c;i++)
        *(A+i) = *(B+i);

}

void mat_copy_int(int *A, int *B, int a, int b, int c) {

    int i;

    for (i=0;i<a*b*c;i++)
        *(A+i) = *(B+i);

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





















