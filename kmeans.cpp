#include <iostream>
#include <armadillo>
#include <cmath>
#include "kmeans.h"



arma::mat defInitCenterPoints(int knum, int scale_) {

    //std::cout << "inside CenterPOints init" << std::endl;
    arma::mat centerPoints(knum, 2, arma::fill::randu);
    centerPoints *= scale_;

    return centerPoints;
}

// Calculates the distance for each 2 points
int calcPointDist(arma::rowvec point_1, arma::rowvec point_2) {
    int sum{ 0 };
    //std::cout << "inside calcdistpoint" << std::endl;

    for (int z{ 0 }; z < point_1.size(); ++z) {
        double diff = point_1[z] - point_2[z];
        diff = pow(diff, 2);
        sum += sqrt(diff);
    }

    return sum;
}


void calcDistanceTotal(arma::mat& initPoints, arma::mat centers) {
    arma::mat centerDist(initPoints.n_rows, centers.n_rows);

    for (int i{ 0 }; i < centers.n_rows; ++i) {
        //center points for loop

        //std::cout << "Distance Total - FIrst loop" << std::endl;
        arma::rowvec curCenter(centers.row(i));
        //curCenter.print();
        for (int j{ 0 }; j < initPoints.n_rows; ++j) {
            centerDist(j, i) = calcPointDist(curCenter, initPoints.row(j));
        }
    }
    //std::cout << "Center Distance:" << std::endl;
    //centerDist.print();

    // find the index of the minimum distance
    arma::ucolvec dists = arma::index_min(centerDist, 1);
    // Convert the data to a format to merge it with the rest of the data
    arma::colvec groupCols = arma::conv_to< arma::colvec >::from(dists);
    //std::cout << "Min Groups: " << std::endl;

    // Add the indeex of the minimum distance to the matrix
    initPoints.col(initPoints.n_cols - 1) = groupCols;

}

void calcNewCentroids(arma::mat pointMat, arma::mat& centroids, int knum) {

    int colNums = pointMat.n_cols;

    // For each group start getting the mean values
    for (int k{ 0 }; k < knum; ++k) {
        arma::rowvec sumVec(colNums, arma::fill::zeros);
        int rowCounter{ 0 };

        for (int i{ 0 }; i < pointMat.n_rows; ++i) {
            double groupNum = pointMat.row(i)[colNums - 1];

            if (groupNum == k) {
                sumVec += pointMat.row(i);
                ++rowCounter;
            }

        }
        // Doing the mean
        sumVec /= rowCounter;
        // Removing the last column since it is the group number
        sumVec = sumVec.subvec(0, colNums - 2);
        //sumVec.print();
        centroids.row(k) = sumVec;
    }
}

bool checkErrImprove(arma::mat prevCenters, arma::mat curCenters) {
    
    prevCenters -= curCenters;
    prevCenters = arma::abs(prevCenters);
    
    // Check to see all the elements have improved less than a specified amount
    bool checkErr = arma::all(arma::vectorise(prevCenters) < 0.0001);
    
    return checkErr;
}