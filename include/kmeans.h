#include <armadillo>
#include "kMeansBase.h"



class kMeans : public kMeansBase {
public:
	kMeans() = default;
    kMeans(int a) : kMeansBase(a){}
	kMeans(int a, int b) : kMeansBase(a, b){}

	kMeans& initCenterPts();

};

kMeans& kMeans::initCenterPts() {

    scaleStandard();

    centerPoints = arma::mat(kNum, 2, arma::fill::zeros);

    // Create a random vector and scale it to the number of points to see which points have got chosen
    arma::rowvec chosNum(kNum, arma::fill::randu);
    chosNum *= points.n_rows;

    // Round up the numbers to get whole numbers
    chosNum = arma::ceil(chosNum);

    // Fill in the data for chosen centers
    for (int i{ 0 }; i < kNum; ++i) {
        centerPoints.row(i) = points.row(chosNum(i));
    }

	return *this;
}