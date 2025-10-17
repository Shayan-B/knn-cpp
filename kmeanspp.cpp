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

arma::uword findMaxDist(arma::colvec dists_, arma::rowvec preCentersIdx) {
	
	arma::uword maxDistIdx{ 0 };
	
	for (int counter{ 0 }; counter < preCentersIdx.n_cols; ++counter) {
		maxDistIdx = arma::index_max(dists_);
		
		if (maxDistIdx == preCentersIdx(counter)) {
			dists_(counter) = 0;
		}
	}

	return maxDistIdx;
}

// Calculates the distance for each 2 points
int calcPointDistpp(arma::rowvec point_1, arma::rowvec point_2) {
	double sum{ 0 };
	//std::cout << "inside calcdistpoint" << std::endl;

	for (int z{ 0 }; z < point_1.size(); ++z) {
		double diff = point_1[z] - point_2[z];
		diff = pow(diff, 2);
		sum += sqrt(diff);
	}

	return sum;
}

arma::mat InitCenterKpp(arma::mat pointMat, int knum) {
	
	arma::mat centers_(knum, 2, arma::fill::zeros);
	arma::rowvec centerIdx(knum, arma::fill::zeros);
	arma::colvec distVec(pointMat.n_rows, arma::fill::zeros);

	
	centerIdx(0) = genRndNum(pointMat.n_rows);
	centers_.row(0) = pointMat.row(centerIdx(0));

	for (int k{ 1 }; k < knum; ++k) {
		arma::rowvec curCenter = centers_.row(k - 1);

		for (int j{ 0 }; j < pointMat.n_rows; ++j) {
			int dist = calcPointDistpp(curCenter, pointMat.row(j));
			distVec(j) = dist;
		}
		arma::uword idxMax = findMaxDist(distVec, centerIdx);
		centers_.row(k) = pointMat.row(idxMax);

	}


	return centers_;
}
