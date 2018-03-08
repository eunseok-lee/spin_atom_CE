#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

int main(int argc, char **argv) {
    
    FILE *fp, *fp1, *fp2, *fp3, *fp4;
    int i, j, k, l, m, n, np1, np2;
    int setn;
    int ndata;
    double *rp;
    double tmp;
    int tmp1, tmp2;
    double tmp3, tmp4;
    char datfilename[100];
    char buff_line[200];
    
    int n1_old = atoi(argv[1]);
    int n2_old = atoi(argv[2]);
    int n3_old = atoi(argv[3]);
    int n1_new = atoi(argv[4]);
    int n2_new = atoi(argv[5]);
    int n3_new = atoi(argv[6]);
    
    struct stat st = {0};

    sprintf(datfilename,"rp_%d_%d_%d.dat", n1_old, n2_old, n3_old);
    if (stat(datfilename, &st) != 0) {
        printf("[E] %s doesn't exist\n",datfilename);
        return(0);
    }
    fp = fopen(datfilename,"r");
    np1 = 0;
    while(fgets(buff_line,sizeof(buff_line),fp) != NULL) {
        if (buff_line[0] == '\n') {
            continue;
        }
        else
            np1++;
    }
    fclose(fp);
    double rp_old[np1][3];
    fp = fopen(datfilename,"r");
    for (i=0;i<np1;i++) {
        fscanf(fp, "%lf", &tmp);
        rp_old[i][0] = tmp;
        fscanf(fp, "%lf", &tmp);
        rp_old[i][1] = tmp;
        fscanf(fp, "%lf", &tmp);
        rp_old[i][2] = tmp;
//        printf("%d\t%f %f %f\n",i,rp_old[i][0],rp_old[i][1],rp_old[i][2]);
    }
    fclose(fp);
    
    sprintf(datfilename,"rp_%d_%d_%d.dat", n1_new, n2_new, n3_new);
    if (stat(datfilename, &st) != 0) {
        printf("[E] %s doesn't exist\n",datfilename);
        return(0);
    }
    fp = fopen(datfilename,"r");
    np2 = 0;
    while(fgets(buff_line,sizeof(buff_line),fp) != NULL) {
        if (buff_line[0] == '\n') {
            continue;
        }
        else
            np2++;
    }
    fclose(fp);
    double rp_new[np2][3];
    fp = fopen(datfilename,"r");
    for (i=0;i<np2;i++) {
        fscanf(fp, "%lf", &tmp);
        rp_new[i][0] = tmp;
        fscanf(fp, "%lf", &tmp);
        rp_new[i][1] = tmp;
        fscanf(fp, "%lf", &tmp);
        rp_new[i][2] = tmp;
//        printf("%d\t%f %f %f\n",i,rp_new[i][0],rp_new[i][1],rp_new[i][2]);
    }
    fclose(fp);
//    printf("np1 = %d, np2 = %d\n",np1,np2);
    
    fp1 = fopen("data_orig.dat","r");
    ndata = 0;
    while(fgets(buff_line,sizeof(buff_line),fp1) != NULL) {
        if (buff_line[0] == '\n') {
            continue;
        }
        else
            ndata++;
    }
    fclose(fp1);
    
    int data[ndata][np1];
    int magmom[ndata][np1];
    double E[ndata];
    double Ef[ndata];
    fp1 = fopen("data_orig.dat","r");
    fp2 = fopen("magmom_orig.dat","r");
    fp3 = fopen("E_orig.dat","r");
    fp4 = fopen("Ef_orig.dat","r");
    for (i=0;i<ndata;i++) {
        for (j=0;j<np1;j++) {
            fscanf(fp1, "%d", &tmp1);
            data[i][j] = tmp1;
            fscanf(fp2, "%d", &tmp2);
            magmom[i][j] = tmp2;
//            printf("data[%d][%d]=%d \t magmom[%d][%d]=%d\n",i,j,data[i][j],i,j,magmom[i][j]);
        }
        fscanf(fp3, "%lf", &tmp3);
        fscanf(fp4, "%lf", &tmp4);
        E[i] = tmp3;
        Ef[i] = tmp4;
    }
    fclose(fp1);
    fclose(fp2);
    fclose(fp3);
    fclose(fp4);
//    printf("ndata = %d\n",ndata);
//    printf("np = %d\n", np1);

    int n1_ext = n1_new/n1_old;
    int n2_ext = n2_new/n2_old;
    int n3_ext = n3_new/n3_old;
    double rp_ext[np2][3];
    int data_ext[ndata][np2];
    int magmom_ext[ndata][np2];
    n = 0;
    for (i=0; i<n1_ext; i++)
        for (j=0; j<n2_ext; j++)
            for (k=0; k<n3_ext; k++)
                for (l=0; l<np1; l++) {
                    rp_ext[n][0] = (i+rp_old[l][0])/n1_ext;
                    rp_ext[n][1] = (j+rp_old[l][1])/n2_ext;
                    rp_ext[n][2] = (k+rp_old[l][2])/n3_ext;
                    for (m=0;m<ndata;m++) {
                        data_ext[m][n] = data[m][l];
                        magmom_ext[m][n] = magmom[m][l];
                    }
//                    printf("rp_ext[%d][:] = %f, %f, %f\n",n,rp_ext[n][0],rp_ext[n][1],rp_ext[n][2]);
                    n++;
                }
//    for (i=0;i<ndata;i++)
//        for (j=0;j<np2;j++)
//            printf("data_ext[%d][%d]=%d \t magmom_ext[%d][%d]=%d\n",i,j,data_ext[i][j],i,j,magmom_ext[i][j]);
//    printf("n = %d\n",n);
    
    int data_new[ndata][np2];
    int magmom_new[ndata][np2];
    double dij[3];
    int id_rp_new_to_ext[np2];
    
    for (i=0;i<np2;i++)
        id_rp_new_to_ext[i] = -1;
    
    for (i=0;i<np2;i++) {
        for (j=0;j<np2;j++) {
            dij[0] = pow(rp_ext[j][0] - rp_new[i][0],2.0);
            dij[1] = pow(rp_ext[j][1] - rp_new[i][1],2.0);
            dij[2] = pow(rp_ext[j][2] - rp_new[i][2],2.0);
            if (sqrt(dij[0]+dij[1]+dij[2]) < 0.01) {
                id_rp_new_to_ext[i] = j;
//                printf("id_rp_new_to_ext[%d] = %d\n", i, id_rp_new_to_ext[i]);
            }
        }
        if (id_rp_new_to_ext[i] < 0)
            printf("[E] failed mapping for rp_new[%d][:]\n",i);
    }
    
    for (i=0;i<ndata;i++)
        for (j=0;j<np2;j++) {
            data_new[i][j] = data_ext[i][id_rp_new_to_ext[j]];
            magmom_new[i][j] = magmom_ext[i][id_rp_new_to_ext[j]];
        }
    
    fp1 = fopen("data_new.dat","w");
    fp2 = fopen("magmom_new.dat","w");
    fp3 = fopen("E_new.dat","w");
    fp4 = fopen("Ef_new.dat","w");
    for (i=0;i<ndata;i++) {
        for (j=0;j<np2;j++) {
            fprintf(fp1, "%d ",data_new[i][j]);
            fprintf(fp2, "%d ",magmom_new[i][j]);
        }
        fprintf(fp1, "\n");
        fprintf(fp2, "\n");
        fprintf(fp3, "%.6f\n",E[i]*n1_ext*n2_ext*n3_ext);
        fprintf(fp4, "%.6f\n",Ef[i]*n1_ext*n2_ext*n3_ext);
    }
    
    printf("\n data_orig and magmom_orig in [n1,n2,n3]=[%d,%d,%d] \n were converted to the ones in [n1,n2,n3]=[%d,%d,%d]\n\n",n1_old,n2_old,n3_old,n1_new,n2_new,n3_new);
    
}


















