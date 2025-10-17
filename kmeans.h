#include <armadillo>

arma::mat defInitCenterPoints(arma::mat initpoints, int knum);
void calcDistanceTotal(arma::mat& initPoints, arma::mat centers);
int calcPointDist(arma::vec point_1, arma::vec point_2);
void calcNewCentroids(arma::mat pointMat, arma::mat& centroids, int knum);
bool checkErrImprove(arma::mat prevCenters, arma::mat curCenters);
