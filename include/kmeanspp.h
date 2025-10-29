#include <armadillo>
#include "kMeansBase.h"



class kMeanspp : public kMeansBase {
public:
    kMeanspp() = default;
    kMeanspp(int a) : kMeansBase(a) {}
    kMeanspp(int a, int b) : kMeansBase(a, b) {}

    kMeanspp& initCenterPts();
    // Utility functions
    arma::uword findMaxDist(arma::colvec, arma::rowvec);
    int genRndNum(int);
};

kMeanspp& kMeanspp::initCenterPts() {

    scaleStandard();

    centerPoints = arma::mat(kNum, 2, arma::fill::zeros);

    arma::rowvec centerIdx(kNum, arma::fill::zeros);
    arma::colvec distVec(points.n_rows, arma::fill::zeros);

    // Choose one point by random
    centerIdx(0) = genRndNum(points.n_rows);
    centerPoints.row(0) = points.row(centerIdx(0));

    // For the rest of the centroids start finding the furthest points to the current one
    for (int k{ 1 }; k < kNum; ++k) {
        arma::rowvec curCenter = centerPoints.row(k - 1);

        for (int j{ 0 }; j < points.n_rows; ++j) {
            int dist = calcPointDist(curCenter, points.row(j));
            distVec(j) = dist;
        }
        arma::uword idxMax = findMaxDist(distVec, centerIdx);
        centerPoints.row(k) = points.row(idxMax);

    }

    return *this;
}

arma::uword kMeanspp::findMaxDist(arma::colvec dists_, arma::rowvec preCentersIdx) {

    arma::uword maxDistIdx{ 0 };

    for (int counter{ 0 }; counter < preCentersIdx.n_cols; ++counter) {

        // Foind the maximum distance
        maxDistIdx = arma::index_max(dists_);

        // Check to see if the chosen point is already in the list of centroids
        if (maxDistIdx == preCentersIdx(counter)) {
            dists_(counter) = 0;
        }
    }

    return maxDistIdx;
}

int kMeanspp::genRndNum(int numRange) {
    std::uniform_int_distribution<size_t> u(0, numRange);
    std::default_random_engine e;
    int rndNum = u(e);
    return rndNum;
}