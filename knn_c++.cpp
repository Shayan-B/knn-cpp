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
void calcDistanceTotal(arma::mat& initPoints, arma::mat centers);
int calcPointDist(arma::vec point_1, arma::vec point_2);
void calcNewCentroids(arma::mat pointMat, arma::mat& centroids, int knum);


int main() {
    // define th epoints
    arma::mat points(10, 2, arma::fill::randu);
    int scaleFactor = 20;
    int kNum = 2;
    int myCounter{ 0 };
    arma::colvec zeroCol(points.n_rows, arma::fill::zeros);

    // scale the points
    points *= scaleFactor;
    //add a columns for group number
    points.insert_cols(points.n_cols, zeroCol);

    arma::mat centerPoints = defInitCenterPoints(kNum, scaleFactor);
    std::cout << "First center points: " << std::endl;
    centerPoints.print();

    std::cout << "Before Calculating distance total" << std::endl;
    calcDistanceTotal(points, centerPoints);

    points.brief_print();
    
    std::cout << "recalcualting centroids" << std::endl;
    calcNewCentroids(points, centerPoints, kNum);


    return 0;
}

arma::mat defInitCenterPoints(int knum, int scale_) {
    
    std::cout << "inside CenterPOints init" << std::endl;
    arma::mat centerPoints(knum, 2, arma::fill::randu);
    centerPoints *= scale_;

    return centerPoints;
}

// Calculates the distance for each 2 points
int calcPointDist(arma::rowvec point_1, arma::rowvec point_2) {
    int sum{ 0 };
    std::cout << "inside calcdistpoint" << std::endl;
    
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
        
        std::cout << "Distance Total - FIrst loop" << std::endl;
        arma::rowvec curCenter(centers.row(i));
        curCenter.print();
        for (int j{ 0 }; j < initPoints.n_rows; ++j) {
            centerDist(j, i) = calcPointDist(curCenter, initPoints.row(j));
        }
    }
    //std::cout << "Center Distance:" << std::endl;
    centerDist.print();

    // find the index of the minimum distance
    arma::ucolvec dists = arma::index_min(centerDist, 1);
    // Convert the data to a format to merge it with the rest of the data
    arma::colvec groupCols = arma::conv_to< arma::colvec >::from(dists);
    std::cout << "Min Groups: " << std::endl;

    // Add the indeex of the minimum distance to the matrix
    initPoints.col(initPoints.n_cols-1) =  groupCols;

}

void calcNewCentroids(arma::mat pointMat, arma::mat& centroids, int knum) {

    int colNums = pointMat.n_cols;

    // For each group start getting the mean values
    for (int k{ 0 }; k < knum; ++k) {
        arma::rowvec sumVec(colNums, arma::fill::zeros);
        int rowCounter{ 0 };

        for (int i{ 0 }; i < pointMat.n_rows; ++i) {
            double groupNum = pointMat.row(i)[colNums-1];
            
            if (groupNum == k) {
                sumVec += pointMat.row(i);
                ++rowCounter;
            }

        }
        sumVec /= rowCounter;
        sumVec = sumVec.subvec(0, colNums - 2);
        sumVec.print();
    }
}