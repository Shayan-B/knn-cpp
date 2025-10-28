#include <armadillo>
#include "kmeans.h"

int genRndNum(int numRange);
arma::uword findMaxDist(arma::colvec dists_, arma::rowvec preCentersIdx);
arma::mat InitCenterKpp(arma::mat pointMat, int knum);
