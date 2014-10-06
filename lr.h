/* ========================================================
 *   Copyright (C) 2014 All rights reserved.
 *   
 *   filename : lr.h
 *   author   : ***
 *   date     : 2014-10-03
 *   info     : 
 * ======================================================== */

#ifndef _LR_H
#define _LR_H

#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
using namespace std;

#define LINE_LEN 256

double ALPHA        = 0.0001;       // the defualt alpha
int M               = 3;            // the number of train instances
int D               = 4;            // the number of dimension
int MAX_ITERS       = 30;           // max iterator numbers
int ALGO            = 1;            // bdg or sgd algorithm

char * train_file   = NULL;         // train file 
char * predict_file = NULL;         // predict file
char * output_file  = NULL;         // output file

vector < vector<double> > train_features;
vector < vector<double> > predict_features;
vector <double> labels;
vector <char*> train_instances;
vector <char*> predict_instances;
vector <double> J_sita, J_sita_dev;
double J_sita_0, J_sita_0_dev;

#endif //LR_H
