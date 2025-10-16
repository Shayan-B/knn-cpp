/*
knn steps:
1. select some random points
2. compute all the distances from all other points
3. determine the groups based on the distances
4. calcualte new center points
*/

#include <iostream>
#include <armadillo>
#include <cmath>

arma::mat defInitCenterPoints(int knum, int scale_);
arma::mat calcDistanceTotal(arma::mat initPoints, arma::mat centers);
int calcPointDist(arma::vec point_1, arma::vec point_2);


int main() {
    // define th epoints
    arma::mat points(10, 2, arma::fill::randu);
    int scaleFactor = 20;
    int kNum = 2;
    int myCounter{ 0 };
    
    // scale the points
    points *= scaleFactor;

    arma::mat centerPoints = defInitCenterPoints(kNum, scaleFactor);
    centerPoints.print();

    std::cout << "For Loop" << std::endl;
    arma::mat result = calcDistanceTotal(points, centerPoints);


    return 0;
}

arma::mat defInitCenterPoints(int knum, int scale_) {
    
    std::cout << "inside loop" << std::endl;
    arma::mat centerPoints(knum, 2, arma::fill::randu);
    centerPoints *= scale_;

    return centerPoints;
}

// Calculates the distance for each 2 points
int calcPointDist(arma::rowvec point_1, arma::rowvec point_2) {
    int sum{ 0 };

    for (int z{ 0 }; z < point_1.size(); ++z) {
        double diff = point_1[z] - point_2[z];
        diff = pow(diff, 2);
        sum += sqrt(diff);
    }
    
    return sum;
}


arma::mat calcDistanceTotal(arma::mat initPoints, arma::mat centers) {
    arma::mat centerDist(initPoints.n_rows, centers.n_rows);

    for (int i{ 0 }; i < centers.n_rows; ++i) {
        //center points for loop
        
        std::cout << "Second loop" << std::endl;
        arma::rowvec curCenter(centers.row(i));
        curCenter.print();
        for (int j{ 0 }; j < initPoints.n_rows; ++j) {
            centerDist(j, i) = calcPointDist(curCenter, initPoints.row(j));
        }
    }
    std::cout << "Center Distance:" << std::endl;
    centerDist.print();

    // find the index of the minimum distance
    arma::ucolvec dists = arma::index_min(centerDist, 1);
    arma::colvec groupCols = arma::conv_to< arma::colvec >::from(dists);
    std::cout << "Min Groups: " << std::endl;

    // Add the indeex of the minimum distance to the matrix
    centerDist.insert_cols(2, groupCols);
    centerDist.print();

    return initPoints;
}