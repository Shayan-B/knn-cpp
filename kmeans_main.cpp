/*
knn steps:
1. select some random points
2. compute all the distances from all other points
3. determine the groups based on the distances
4. calcualte new center points
5. DO this as many times as possible so that the points won't move anymore
*/

#include <iostream>
#include <armadillo>
#include <cmath>
#include "kmeans.h"




int main() {
    // define the points
    arma::mat points(1000, 2, arma::fill::randu);
    int scaleFactor = 50;
    int kNum = 5;
    int myCounter{ 0 };
    arma::colvec zeroCol(points.n_rows, arma::fill::zeros);

    // scale the points
    points *= scaleFactor;
    // Choose the centerpoints
    arma::mat centerPoints = defInitCenterPoints(points, kNum);

    //add a columns for group number
    points.insert_cols(points.n_cols, zeroCol);
    
    std::cout << "First center points: " << std::endl;
    centerPoints.print();
    for (int counter{0}; counter<100; ++counter){
    
        //std::cout << "Before Calculating distance total" << std::endl;
        calcDistanceTotal(points, centerPoints);

        //points.brief_print();
        arma::mat tempCenters = centerPoints;
        //std::cout << "recalcualting centroids" << std::endl;
        calcNewCentroids(points, centerPoints, kNum);
        //std::cout << "Final centroids" << std::endl;
        //centerPoints.print();

        // Check to sse how much we have improved
        bool checkImprov = checkErrImprove(tempCenters, centerPoints);

        if (checkImprov == true) {
            std::cout << "after " << counter << " iterations We have reachedd a point that we are not improving anymore, quitting program." << std::endl;
            std::cout << "The last center points are: " << std::endl;
            centerPoints.print();
            break;
        }

    }

    return 0;
}

