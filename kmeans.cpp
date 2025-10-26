#include <iostream>
#include <armadillo>
#include <cmath>
#include "kmeans.h"

// Implementation of standard scaler
void scaleStandard(arma::mat& pointMat) {
    
    arma::mat tempPoints = pointMat;
    arma::rowvec meanVec(2, arma::fill::zeros);
   
    // Sum of the points coordinates by column
    meanVec = arma::sum(pointMat, 0);
    meanVec /= pointMat.n_rows;

    // Calculate the standard deviation
    for (int i{ 0 }; i < 2; ++i) {
        tempPoints.col(i) = tempPoints.col(i) - meanVec(i);
    }

    // Element-wise square
    tempPoints = arma::square(tempPoints);
    arma::rowvec stdVec = arma::sum(tempPoints, 0);
    stdVec /= pointMat.n_rows;
    stdVec = arma::sqrt(stdVec);

    for (int i{ 0 }; i < 2; ++i) {
        pointMat.col(i) = (pointMat.col(i) - meanVec(i)) / stdVec(i);
    }
}

// Define the centroids using the normal random initialization
arma::mat defInitCenterPoints(arma::mat initpoints, int knum) {


    arma::mat centerPoints(knum, 2, arma::fill::zeros);
    
    // Create a random vector and scale it to the number of points to see which points have got chosen
    arma::rowvec chosNum(knum, arma::fill::randu);
    chosNum *= initpoints.n_rows;

    // Round up the numbers to get whole numbers
    chosNum = arma::ceil(chosNum);
    
    // Fill in the data for chosen centers
    for (int i{ 0 }; i < knum; ++i) {
        centerPoints.row(i) = initpoints.row(chosNum(i));
    }

    return centerPoints;
}

// Calculates the distance for each 2 points
int calcPointDist(arma::rowvec point_1, arma::rowvec point_2) {
    double sum{ 0 };
    

    for (int z{ 0 }; z < point_1.size(); ++z) {
        double diff = point_1[z] - point_2[z];
        diff = pow(diff, 2);
        sum += sqrt(diff);
    }

    return sum;
}

// Calculates the Total distances of all points an each centroid
void calcDistanceTotal(arma::mat& initPoints, arma::mat centers) {
    arma::mat centerDist(initPoints.n_rows, centers.n_rows);

    //center points for loop
    for (int i{ 0 }; i < centers.n_rows; ++i) {
        arma::rowvec curCenter(centers.row(i));
        
        for (int j{ 0 }; j < initPoints.n_rows; ++j) {
            centerDist(j, i) = calcPointDist(curCenter, initPoints.row(j));
        }
    }

    // Find the index of the minimum distance
    arma::ucolvec dists = arma::index_min(centerDist, 1);
    
    // Convert the data to a format to merge it with the rest of the data
    arma::colvec groupCols = arma::conv_to< arma::colvec >::from(dists);

    // Add the index of the minimum distance to the matrix
    initPoints.col(initPoints.n_cols - 1) = groupCols;

}

// Calculate the new centroids based on the groupings
void calcNewCentroids(arma::mat pointMat, arma::mat& centroids, int knum) {

    // Number of columns - SInce we have and extra column for the KGroup
    double colNums = pointMat.n_cols;

    // For each group start getting the mean values
    for (int k{ 0 }; k < knum; ++k) {
        // Define a vector for keeping the sums and mean
        arma::rowvec sumVec(colNums, arma::fill::zeros);
        
        // We define the row counter at 11 so that we don't get any Division by zeros later
        int rowCounter{ 1 };

        for (int i{ 0 }; i < pointMat.n_rows; ++i) {
            double groupNum = pointMat.row(i)[colNums - 1];

            if (groupNum == k) {
                // summin gup all the values for each point to get the mean
                sumVec += pointMat.row(i);
                if (i != 0) {
                    ++rowCounter;
                }
            }

        }
        
        // Doing the mean
        sumVec /= rowCounter;
        
        // Removing the last column since it is the group number
        sumVec = sumVec.subvec(0, colNums - 2);
        
        // This if prevents adding a zero coordinate for the centroids that have no subset
        if (!sumVec.is_zero()) {
            centroids.row(k) = sumVec;
        }
    }
}

bool checkErrImprove(arma::mat prevCenters, arma::mat curCenters) {
    
    prevCenters -= curCenters;
    prevCenters = arma::abs(prevCenters);
    
    // Check to see all the elements have improved less than a specified amount
    bool checkErr = arma::all(arma::vectorise(prevCenters) < 0.0001);
    
    return checkErr;
}