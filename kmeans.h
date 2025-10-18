#include <armadillo>

void scaleStandard(arma::mat& pointMat);
arma::mat defInitCenterPoints(arma::mat initpoints, int knum);
void calcDistanceTotal(arma::mat& initPoints, arma::mat centers);
int calcPointDist(arma::rowvec point_1, arma::rowvec point_2);
void calcNewCentroids(arma::mat pointMat, arma::mat& centroids, int knum);
bool checkErrImprove(arma::mat prevCenters, arma::mat curCenters);
