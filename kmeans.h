#include <armadillo>

arma::mat defInitCenterPoints(int knum, int scale_);
void calcDistanceTotal(arma::mat& initPoints, arma::mat centers);
int calcPointDist(arma::vec point_1, arma::vec point_2);
void calcNewCentroids(arma::mat pointMat, arma::mat& centroids, int knum);
