#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

int main(int argc, char **argv) {
    
    FILE *fp1, *fp2, *fp3, *fp4, *fp5, *fp6;
    int i, j, k, np, npL, npM, npN;
    int setn, ctr;
    int ndata;
    double *rp;
    double tmp;
    int tmp1;
    double tmp2, tmp3, tmp4, tmp5;
    char dirname[100];
    char datfilename[100];
    char buff_line[200];

    double max_dist_cutoff = 0.08;
    double ind_dist_cutoff = 0.001;
    double dist_cutoff;
    double dij[3];
    double dist;
    
    double magLi = atof(argv[1]);
    double magMn = atof(argv[2]);
    double magNi = atof(argv[3]);
    
    for (i=1;i<100000;i++) {
        sprintf(dirname,"data%d",i);
        struct stat st = {0};
        if (stat(dirname, &st) == -1)
            break;
    }
    ndata = i-1;
    
    printf("ndata = %d\n",ndata);
    fp1 = fopen("rp.dat","r");
    np = 0;
    while(fgets(buff_line,sizeof(buff_line),fp1) != NULL) {
        if (buff_line[0] == '\n') {
            continue;
        }
        else
            np++;
    }
    fclose(fp1);
    printf("np = %d\n", np);
    printf("magLi = %f\n",magLi);
    printf("magMn = %f\n",magMn);
    printf("magNi = %f\n",magNi);

    rp = (double*) calloc (np*3,sizeof(double));
    printf("rp data is read\n");
    fp1 = fopen("rp.dat","r");
    for (i=0;i<np;i++) {
        fscanf(fp1, "%lf", &tmp);
        *(rp+3*i) = tmp;
        fscanf(fp1, "%lf", &tmp);
        *(rp+3*i+1) = tmp;
        fscanf(fp1, "%lf", &tmp);
        *(rp+3*i+2) = tmp;
        printf("%d\t%f %f %f\n",i,*(rp+3*i),*(rp+3*i+1),*(rp+3*i+2));
    }
    fclose(fp1);
    printf("----------------------------------------\n");
    
    int data[ndata][np];
    int magmom[ndata][np];
    int nL[ndata];
    int nM[ndata];
    int nN[ndata];
    double E[ndata];
    double Ef[ndata];
    double *rpL, *rpM, *rpN;
    int *Lid, *Mid, *Nid;
    
    // fill up data and magmom
    for (i=0;i<ndata;i++)
        for (j=0;j<np;j++) {
            data[i][j] = -2; //default, also value for Va
            magmom[i][j] = 0;   //default, also value for Va
        }

    for (setn=0;setn<ndata;setn++) {
        printf("data%d is being processed\n",setn+1);
        sprintf(datfilename,"data%d/posL.dat",setn+1);
        fp2 = fopen(datfilename,"r");
        sprintf(datfilename,"data%d/posM.dat",setn+1);
        fp3 = fopen(datfilename,"r");
        sprintf(datfilename,"data%d/posN.dat",setn+1);
        fp4 = fopen(datfilename,"r");
        
        npL = 0;
        while(fgets(buff_line,sizeof(buff_line),fp2) != NULL) {
            if (buff_line[0] == '\n') {
                continue;
            }
            else
                npL++;
        }
        printf("npL = %d\n", npL);
        npM = 0;
        while(fgets(buff_line,sizeof(buff_line),fp3) != NULL) {
            if (buff_line[0] == '\n') {
                continue;
            }
            else
                npM++;
        }
        printf("npM = %d\n", npM);
        npN = 0;
        while(fgets(buff_line,sizeof(buff_line),fp4) != NULL) {
            if (buff_line[0] == '\n') {
                continue;
            }
            else
                npN++;
        }
        printf("npN = %d\n", npN);
        fclose(fp2);
        fclose(fp3);
        fclose(fp4);
    
        rpL = (double*) calloc(npL*3,sizeof(double));
        rpM = (double*) calloc(npM*3,sizeof(double));
        rpN = (double*) calloc(npN*3,sizeof(double));

        Lid = (int*) malloc(npL*sizeof(int));
        Mid = (int*) malloc(npM*sizeof(int));
        Nid = (int*) malloc(npN*sizeof(int));
        for (i=0;i<npL;i++)
            *(Lid+i) = -1;
        for (i=0;i<npM;i++)
            *(Mid+i) = -1;
        for (i=0;i<npN;i++)
            *(Nid+i) = -1;
        
        printf("posL data is read\n");
        sprintf(datfilename,"data%d/posL.dat",setn+1);
        fp2 = fopen(datfilename,"r");
        for (i=0;i<npL;i++) {
            fscanf(fp2, "%lf", &tmp);
            *(rpL+3*i) = tmp;
            fscanf(fp2, "%lf", &tmp);
            *(rpL+3*i+1) = tmp;
            fscanf(fp2, "%lf", &tmp);
            *(rpL+3*i+2) = tmp;
//            printf("%d\t%f %f %f\n",i,*(rpL+3*i),*(rpL+3*i+1),*(rpL+3*i+2));
        }

        printf("posM data is read\n");
        sprintf(datfilename,"data%d/posM.dat",setn+1);
        fp3 = fopen(datfilename,"r");
        for (i=0;i<npM;i++) {
            fscanf(fp3, "%lf", &tmp);
            *(rpM+3*i) = tmp;
            fscanf(fp3, "%lf", &tmp);
            *(rpM+3*i+1) = tmp;
            fscanf(fp3, "%lf", &tmp);
            *(rpM+3*i+2) = tmp;
//            printf("%d\t%f %f %f\n",i,*(rpM+3*i),*(rpM+3*i+1),*(rpM+3*i+2));
        }
    
        printf("posN data is read\n");
        sprintf(datfilename,"data%d/posN.dat",setn+1);
        fp4 = fopen(datfilename,"r");
        for (i=0;i<npN;i++) {
            fscanf(fp4, "%lf", &tmp);
            *(rpN+3*i) = tmp;
            fscanf(fp4, "%lf", &tmp);
            *(rpN+3*i+1) = tmp;
            fscanf(fp4, "%lf", &tmp);
            *(rpN+3*i+2) = tmp;
//            printf("%d\t%f %f %f\n",i,*(rpN+3*i),*(rpN+3*i+1),*(rpN+3*i+2));
        }
        fclose(fp4);
    
        for (i=0;i<npL;i++) {
            dist_cutoff = ind_dist_cutoff;
            ctr = 0;
            while (dist_cutoff < max_dist_cutoff && Lid[i] < 0) {
                for (j=0;j<np;j++) {
                    dij[0] = pow(fmod(*(rp+3*j+0) - *(rpL+3*i+0) + 0.5, 1.0) - 0.5, 2.0);
                    dij[1] = pow(fmod(*(rp+3*j+1) - *(rpL+3*i+1) + 0.5, 1.0) - 0.5, 2.0);
                    dij[2] = pow(fmod(*(rp+3*j+2) - *(rpL+3*i+2) + 0.5, 1.0) - 0.5, 2.0);
                    dist = sqrt(dij[0] + dij[1] + dij[2]);
                    if (dist < dist_cutoff) {
                        Lid[i] = j;
                        ctr = 1;
                        break;
                    }
                    dist_cutoff = dist_cutoff+ind_dist_cutoff;
                }
            }
            if (ctr == 0)
                printf("[E] Lid[%d] not identified\n",i);
            else
                printf("Lid[%d] = %d\n",i,Lid[i]);
        }
        for (i=0;i<npM;i++) {
            dist_cutoff = ind_dist_cutoff;
            ctr = 0;
            while (dist_cutoff < max_dist_cutoff && Mid[i] < 0) {
                for (j=0;j<np;j++) {
                    dij[0] = pow(fmod(*(rp+3*j+0) - *(rpM+3*i+0) + 0.5, 1.0) - 0.5, 2.0);
                    dij[1] = pow(fmod(*(rp+3*j+1) - *(rpM+3*i+1) + 0.5, 1.0) - 0.5, 2.0);
                    dij[2] = pow(fmod(*(rp+3*j+2) - *(rpM+3*i+2) + 0.5, 1.0) - 0.5, 2.0);
                    dist = sqrt(dij[0] + dij[1] + dij[2]);
                    if (dist < dist_cutoff) {
                        Mid[i] = j;
                        ctr = 1;
                        break;
                    }
                    dist_cutoff = dist_cutoff+ind_dist_cutoff;
                }
            }
            if (ctr == 0)
                printf("[E] Mid[%d] not identified\n",i);
            else
                printf("Mid[%d] = %d\n",i,Mid[i]);
        }
        for (i=0;i<npN;i++) {
            dist_cutoff = ind_dist_cutoff;
            ctr = 0;
            while (dist_cutoff < max_dist_cutoff && Nid[i] < 0) {
                for (j=0;j<np;j++) {
                    dij[0] = pow(fmod(*(rp+3*j+0) - *(rpN+3*i+0) + 0.5, 1.0) - 0.5, 2.0);
                    dij[1] = pow(fmod(*(rp+3*j+1) - *(rpN+3*i+1) + 0.5, 1.0) - 0.5, 2.0);
                    dij[2] = pow(fmod(*(rp+3*j+2) - *(rpN+3*i+2) + 0.5, 1.0) - 0.5, 2.0);
                    dist = sqrt(dij[0] + dij[1] + dij[2]);
                    if (dist < dist_cutoff) {
                        Nid[i] = j;
                        ctr = 1;
                        break;
                    }
                    dist_cutoff = dist_cutoff+ind_dist_cutoff;
                }
            }
            if (ctr == 0)
                printf("[E] Nid[%d] not identified\n",i);
            else
                printf("Nid[%d] = %d\n",i,Nid[i]);
        }

        for (i=0;i<npL;i++)
            data[setn][Lid[i]] =  2;
        for (i=0;i<npM;i++)
            data[setn][Mid[i]] =  1;
        for (i=0;i<npN;i++)
            data[setn][Nid[i]] = -1;
        sprintf(datfilename,"data%d/mag_val.dat",setn+1);
        fp5 = fopen(datfilename,"r");
        for (i=0;i<npL+npM+npN;i++) {
            fscanf(fp5, "%d %lf %lf %lf %lf", &tmp1, &tmp2, &tmp3, &tmp4, &tmp5);
            printf("%d-atom, %f\n",i,tmp5);
            if (i < npL)
                if (tmp5 > magLi)
                    magmom[setn][Lid[i]] = 1;
                else if (tmp5 < -1*magLi)
                    magmom[setn][Lid[i]] = -1;
                else
                    magmom[setn][Lid[i]] = 0;
            else if (i < npL+npM)
                if (tmp5 > magMn)
                    magmom[setn][Mid[i-npL]] = 1;
                else if (tmp5 < -1*magMn)
                    magmom[setn][Mid[i-npL]] = -1;
                else
                    magmom[setn][Mid[i-npL]] = 0;
            else
                if (tmp5 > magNi)
                    magmom[setn][Nid[i-npL-npM]] = 1;
                else if (tmp5 < -1*magNi)
                    magmom[setn][Nid[i-npL-npM]] = -1;
                else
                    magmom[setn][Nid[i-npL-npM]] = 0;
        }
        nL[setn] = npL;
        nM[setn] = npM;
        nN[setn] = npN;

        sprintf(datfilename,"data%d/energy.dat",setn+1);
        fp6 = fopen(datfilename,"r");
        fscanf(fp6, "%d %lf %lf %lf %lf", &tmp1, &tmp2, &tmp3, &tmp4, &tmp5);
        E[setn] = tmp2;
        printf("E[%d] = %f\n",setn,E[setn]);
//        printf("data[%d][1:4] = %d %d %d %d\n",setn,data[setn][0],data[setn][1],data[setn][2],data[setn][3]);
//        printf("magmom[%d][:] = %d %d %d %d\n",setn,magmom[setn][0],magmom[setn][1],magmom[setn][2],magmom[setn][3]);

        fclose(fp2);
        fclose(fp3);
        fclose(fp4);
        fclose(fp5);
        fclose(fp6);
    }
    
    // identify the energy at end states
    double EM, EN;
    int fullc;
    fullc = (int) np/2;
    for (i=0;i<ndata;i++) {
        if (nM[i] == fullc && nL[i] == fullc && E[i] < EM)
            EM = E[i];
        if (nN[i] == fullc && nL[i] == fullc && E[i] < EN)
            EN = E[i];
    }
    for (i=0;i<ndata;i++)
        Ef[i] = E[i] - nM[i]*EM/(np*1.0/2) - nN[i]*EN/(np*1.0/2);

    printf("----------------------------------------\n");
    printf("Data Table\n");
    for (i=0;i<ndata;i++) {
        for (j=0;j<np;j++)
            printf("%2d ",data[i][j]);
        printf("\n");
    }
    printf("----------------------------------------\n");
    printf("Magmom Table\n");
    for (i=0;i<ndata;i++) {
        for (j=0;j<np;j++)
            printf("%2d ",magmom[i][j]);
        printf("\n");
    }
    printf("----------------------------------------\n");
    printf("Ef\n");
    for (i=0;i<ndata;i++)
        printf("%.4e\n",Ef[i]);
    
    FILE *fp7, *fp8, *fp9, *fp10;
    fp7 = fopen("data_orig.dat","w");
    fp8 = fopen("magmom_orig.dat","w");
    fp9 = fopen("E_orig.dat","w");
    fp10= fopen("Ef_orig.dat","w");
    for (i=0;i<ndata;i++) {
        for (j=0;j<np;j++) {
            fprintf(fp7, "%d ",data[i][j]);
            fprintf(fp8, "%d ",magmom[i][j]);
        }
        fprintf(fp7, "\n");
        fprintf(fp8, "\n");
        fprintf(fp9, "%.6f\n",E[i]);
        fprintf(fp10,"%.6f\n",Ef[i]);
    }
}


















