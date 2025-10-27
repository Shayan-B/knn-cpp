#include <armadillo>
#include "kmeans.h"
#include "kmeanspp.h"

import <cmath>;
import <string>;
import <iostream>;

using std::string;


int main() {
    // define the points
    arma::mat points(1000, 2, arma::fill::randu);
    int scaleFactor = 50;
    int kNum = 6;
    int myCounter = 0;

    // This Variable can be either kmeans or kmeans++ and defines the center initialization
    string initializeMethod = "kmeans++";
    
    
    arma::colvec zeroCol(points.n_rows, arma::fill::zeros);
    
    // scale the points
    points *= scaleFactor;

    // Apply Standard Scaler
    scaleStandard(points);
    points.brief_print();

    // Choose the centerpoints - kmeans or kmeans++ initialization
    arma::mat centerPoints = (initializeMethod == "kmeans++") ? InitCenterKpp(points, kNum) : defInitCenterPoints(points, kNum);

    //add a column for group number K
    points.insert_cols(points.n_cols, zeroCol);
    
    std::cout << "First center points: " << std::endl;
    centerPoints.print();
    std::cout << "-------------------" << std::endl;

    for (int counter{0}; counter<50; ++counter){
    
        // Calculate the total distances of each point and all of the centroids
        calcDistanceTotal(points, centerPoints);

        // Keep a copy of the centroids before changing in order to calculate the amount of change
        arma::mat tempCenters = centerPoints;
        
        // Recalculate the new Centeroids
        calcNewCentroids(points, centerPoints, kNum);
        
        centerPoints.print();
        std::cout << "-----------" << std::endl;

        // Check to see how much we have improved
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

