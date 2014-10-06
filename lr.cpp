/* ========================================================
 *   Copyright (C) 2014 All rights reserved.
 *   
 *   filename : lr.cpp
 *   author   : guoxinpeng 
 *   date     : 2014-10-03
 *   info     : linear regression
 * ======================================================== */

#include "lr.h"

double logistic_func(double k) {
    return 1.0/(1.0+exp(-k));
}

int command_line_parse(int argc, char * argv[]) {
    int i = 0;
    while (i < argc) {
        char * arg = argv[i];
        if (strcmp(arg,"-n") == 0) { MAX_ITERS = atoi(argv[++i]); }
        else if (strcmp(arg, "-t") == 0) { train_file = argv[++i]; }
        else if (strcmp(arg, "-p") == 0) { predict_file = argv[++i]; }
        else if (strcmp(arg, "-o") == 0) { output_file= argv[++i]; }
        else if (strcmp(arg, "-a") == 0) { ALPHA = atof(argv[++i]); }
        else if (strcmp(arg, "-k") == 0) { ALGO  = atoi(argv[++i]); } 
        i++;
    }
    if (predict_file == NULL || train_file == NULL || output_file == NULL) {
        fprintf(stderr, "ERROR: command line not well formatted!");
        return -1;
    }
    return 0;
}

void print_help() {
    fprintf(stderr, "\n\n     Linear Regression Command Usage:    \n");
    fprintf(stderr, "\n     ./lr -n <int> -a <double> -k <int> -t <string> -i <string> -o <string>\n");
    fprintf(stderr, "\n     -n maximum iterators                ");
    fprintf(stderr, "\n     -a alpha parameter                  ");
    fprintf(stderr, "\n     -k [1,2] bgd or sgd algorithm       ");
    fprintf(stderr, "\n     -t train file                       ");
    fprintf(stderr, "\n     -p predict file                     ");
    fprintf(stderr, "\n     -o output file                      \n\n");
}

double func(int k) {
    double ret = 0;
    for(int i = 0; i < D; i++) ret += train_features[k][i]*J_sita[i];
    return logistic_func(ret+J_sita_0)-labels[k];
}

void bdg() {
    for(int j = 0; j < D; j++) {
        J_sita_dev[j] = 0;
        for(int i = 0; i < M; i++) {
            J_sita_dev[j] += func(i)*train_features[i][j];
        }
    }
    J_sita_0_dev = 0;
    for(int i = 0; i < M; i++) J_sita_0_dev += func(i); 
    
    for(int j = 0; j < D; j++) J_sita_dev[j] /= M;
    J_sita_0_dev /= M;
    
    for(int j = 0; j < D; j++) J_sita[j] -= ALPHA*J_sita_dev[j];
    J_sita_0 -= ALPHA*J_sita_0_dev;
}

void sgd(int k) {
    for(int j = 0; j < D; j++) {
        J_sita_dev[j] = func(k)*train_features[k][j];
    }
    J_sita_0_dev = func(k);

    for(int j = 0; j < D; j++) J_sita[j] -= ALPHA*J_sita_dev[j];
    J_sita_0 -= ALPHA*J_sita_0_dev;
}

char ** split(const char * string, char delim, int * count) {
    if( !string )
        return 0;
    int i, j, c;
    i = 0; j = c = 1;
    int length = strlen(string);
    char * copy_str = (char *) malloc(length+1);
    memmove(copy_str, string, length);
    copy_str[length] = '\0';
    for(; i < length; i++) {
        if(copy_str[i] == delim) {
            c += 1;
        }
    }
    (*count) = c;
    char** str_array = (char**)malloc(sizeof(char*) * c);
    str_array[0] = copy_str;
    for(i = 0; i < length; i++) {
        if(copy_str[i] == delim) {
            copy_str[i] = '\0';
            str_array[j++] = copy_str + i + 1;
        }
    }
    return str_array;
}

int load_data() {
    FILE * tfp, * pfp;
    int line = 0, sz = 0; 
    char buffer[LINE_LEN];
    char ** str_array = NULL;
 
    // load train file
    if((tfp = fopen(train_file, "r")) == NULL) {
        fprintf(stderr, "train file is not valid\n");
        return -1;
    }
    M = 0;
    while(fgets(buffer, LINE_LEN, tfp) != NULL) {
        vector<double> tmp;
        str_array = split(buffer, '\t', &sz);
        D = sz - 2;
        train_instances.push_back(str_array[0]);
        for(int i = 1; i < sz-1; i++) {
            tmp.push_back(atof(str_array[i]));
        }
        train_features.push_back(tmp);
        labels.push_back(atof(str_array[sz-1]));
        M++;
    }
    fclose(tfp);

    // load predict data
    if((pfp = fopen(predict_file, "r")) == NULL) {
        fprintf(stderr, "load predict file failed");
        return -1;
    }
    while(fgets(buffer, LINE_LEN, pfp) != NULL) {
        vector<double> tmp;
        str_array = split(buffer, '\t', &sz);
        predict_instances.push_back(str_array[0]);
        for(int i = 1; i < sz; i++) {
            tmp.push_back(atof(str_array[i]));
        }
        predict_features.push_back(tmp);
    }
    fclose(pfp);
    return 0;
}

void init_alldata() {
    for(int i = 0; i < D; i++) { J_sita.push_back(0.0);  J_sita_dev.push_back(0.0); }
    J_sita_0 = 0.0;
}

void sgd_train_model() {
    for(int iter = 0; iter < MAX_ITERS; iter++) {
        sgd(iter%M);
    }
}

void bgd_train_model() {
    for(int iter = 0; iter < MAX_ITERS; iter++) {
        bdg();
    }
}

double pfunc(int k) {
    double ret = 0.0;
    for(int i = 0; i < D; i++) {
        ret += J_sita[i]*predict_features[k][i];
    }
    return logistic_func(ret+J_sita_0) > 0.5 ? 1.0 : 0.0;
}

void predict() {
    FILE * ofp;
    if((ofp = fopen(output_file, "w")) == NULL) {
        fprintf(stderr, "ERROR: Can't open output file");
    }
    for(int i = 0; i < predict_features.size(); i++) {
        fprintf(ofp, "%s", predict_instances[i]);
        for(int j = 0; j < D; j++) {
            fprintf(ofp, "\t%.3f", predict_features[i][j]);
        }
        fprintf(ofp, "\t%.3f\n", pfunc(i));
    }
    fclose(ofp);
}

int main(int argc, char * argv[]) {
    if (command_line_parse(argc, argv) != 0) { print_help(); return -1; }
    if (load_data() != 0) { print_help(); return -1; }
    init_alldata();
    if(ALGO == 1) bgd_train_model();
    else sgd_train_model();
    predict();
    return 0;
}
