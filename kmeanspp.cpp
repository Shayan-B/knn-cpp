#include <iostream>
#include <armadillo>
#include <random>
#include "kmeanspp.h"
#include "kmeans.h"

int genRndNum(int numRange) {
	std::uniform_int_distribution<size_t> u(0, numRange);
	std::default_random_engine e;
	int rndNum = u(e);
	return rndNum;
}

// Find the point with the maximum distance to the current point, used in kmeans++ innitialization
arma::uword findMaxDist(arma::colvec dists_, arma::rowvec preCentersIdx) {
	
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

// Initialize Center points based on the Kmeans++ algorithm
arma::mat InitCenterKpp(arma::mat pointMat, int knum) {
	
	arma::mat centers_(knum, 2, arma::fill::zeros);
	arma::rowvec centerIdx(knum, arma::fill::zeros);
	arma::colvec distVec(pointMat.n_rows, arma::fill::zeros);

	// Choose one point by random
	centerIdx(0) = genRndNum(pointMat.n_rows);
	centers_.row(0) = pointMat.row(centerIdx(0));

	// For the rest of the centroids start finding the furthest points to the current one
	for (int k{ 1 }; k < knum; ++k) {
		arma::rowvec curCenter = centers_.row(k - 1);

		for (int j{ 0 }; j < pointMat.n_rows; ++j) {
			int dist = calcPointDist(curCenter, pointMat.row(j));
			distVec(j) = dist;
		}
		arma::uword idxMax = findMaxDist(distVec, centerIdx);
		centers_.row(k) = pointMat.row(idxMax);

	}


	return centers_;
}
