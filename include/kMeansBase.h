import <iostream>;
import <cmath>;
import <string>;
#include <armadillo>
#include <format>

class kMeansBase {
	friend std::ostream& operator<<(std::ostream&, const kMeansBase&);

public:

	// Constructors
	kMeansBase() = default;
	kMeansBase(int numPts) : numPoints(numPts) {
		std::cout << "you only initialized the number of points in the database, The number of Groups (K) will be set to 2\n"
			<< "If you want to set the number of groups manually you can call the class again or use the setNumK method";
	}
	kMeansBase(int numK, int numPts): kNum(numK), numPoints(numPts){
		points = arma::mat(numPoints, 2, arma::fill::zeros);
		centerPoints = arma::mat(kNum, 2, arma::fill::zeros);
	}
	int calcPointDist(const arma::rowvec&, const arma::rowvec&);

	kMeansBase& genPoints();

	// Initialization of center points - will be different in each of the child classes
	virtual kMeansBase& initCenterPts();
	kMeansBase& setNumK(int);
	kMeansBase& calcDistanceTotal();

	// Main function to call
	kMeansBase& searchCenters();

	// Utility Functions

	// Implementation of standard scaler
	void scaleStandard();
	void calcNewCentroids();
	bool checkErrImprove(const arma::mat);

protected:
	std::string modelName{ "Base" };
	int numPoints{ 0 };
	int kNum{ 2 };
	int scaleFactor{ 50 };
	arma::mat points;
	arma::mat centerPoints;
};



// Generate the data points
kMeansBase& kMeansBase::genPoints() {
	
	points = arma::mat(numPoints, 2, arma::fill::randu);
	// scale the points
	points *= scaleFactor;

	return *this;
}

// Generate the initialization points - depends on the algorithm
kMeansBase& kMeansBase::initCenterPts() {
	return *this;
}

// Set the number of groups
kMeansBase& kMeansBase::setNumK(int numK) {
	this->kNum = numK;
	return *this;
}

void kMeansBase::scaleStandard() {

	arma::mat tempPoints = points;
	arma::rowvec meanVec(2, arma::fill::zeros);

	// Sum of the points coordinates by column
	meanVec = arma::sum(points, 0);
	meanVec /= points.n_rows;

	// Calculate the standard deviation
	for (int i{ 0 }; i < 2; ++i) {
		tempPoints.col(i) = tempPoints.col(i) - meanVec(i);
	}

	// Element-wise square
	tempPoints = arma::square(tempPoints);
	arma::rowvec stdVec = arma::sum(tempPoints, 0);
	stdVec /= points.n_rows;
	stdVec = arma::sqrt(stdVec);

	for (int i{ 0 }; i < 2; ++i) {
		points.col(i) = (points.col(i) - meanVec(i)) / stdVec(i);
	}

}

int kMeansBase::calcPointDist(const arma::rowvec& point_1, const arma::rowvec& point_2) {
	double sum{ 0 };


	for (int z{ 0 }; z < point_1.size(); ++z) {
		double diff = point_1[z] - point_2[z];
		diff = pow(diff, 2);
		sum += sqrt(diff);
	}

	return sum;
}

kMeansBase& kMeansBase::calcDistanceTotal() {

	arma::mat centerDist(points.n_rows, centerPoints.n_rows);

	//center points for loop
	for (int i{ 0 }; i < centerPoints.n_rows; ++i) {
		arma::rowvec curCenter(centerPoints.row(i));

		for (int j{ 0 }; j < points.n_rows; ++j) {
			centerDist(j, i) = calcPointDist(curCenter, points.row(j));
		}
	}

	// Find the index of the minimum distance
	arma::ucolvec dists = arma::index_min(centerDist, 1);

	// Convert the data to a format to merge it with the rest of the data
	arma::colvec groupCols = arma::conv_to< arma::colvec >::from(dists);

	// Add the index of the minimum distance to the matrix
	points.col(points.n_cols - 1) = groupCols;

	return *this;

}


kMeansBase& kMeansBase::searchCenters() {
	genPoints();
	initCenterPts();

	// Insert col for K Group Number
	arma::colvec zeroCol(points.n_rows, arma::fill::zeros);
	points.insert_cols(points.n_cols, zeroCol);

	for (int counter{ 0 }; counter < 50; ++counter) {

		// Calculate the total distances of each point and all of the centroids
		calcDistanceTotal();

		// Keep a copy of the centroids before changing in order to calculate the amount of change
		arma::mat tempCenters = centerPoints;

		// Recalculate the new Centeroids
		calcNewCentroids();

		centerPoints.print();
		std::cout << "-----------" << std::endl;

		// Check to see how much we have improved
		bool checkImprov = checkErrImprove(tempCenters);

		if (checkImprov == true) {
			std::cout << "after " << counter << " iterations We have reachedd a point that we are not improving anymore, quitting program." << std::endl;
			std::cout << "The last center points are: " << std::endl;
			centerPoints.print();
			break;
		}

	}

	return *this;
}

void kMeansBase::calcNewCentroids() {

	// Number of columns - SInce we have and extra column for the KGroup
	double colNums = points.n_cols;

	// For each group start getting the mean values
	for (int k{ 0 }; k < kNum; ++k) {
		// Define a vector for keeping the sums and mean
		arma::rowvec sumVec(colNums, arma::fill::zeros);

		// We define the row counter at 11 so that we don't get any Division by zeros later
		int rowCounter{ 1 };

		for (int i{ 0 }; i < points.n_rows; ++i) {
			double groupNum = points.row(i)[colNums - 1];

			if (groupNum == k) {
				// summin gup all the values for each point to get the mean
				sumVec += points.row(i);
				if (i != 0) {
					++rowCounter;
				}
			}

		}

		// Doing the mean
		sumVec /= rowCounter;

		// Removing the last column since it is the group number
		sumVec = sumVec.subvec(0, colNums - 2);

		// This if prevents adding a zero coordinate for the centerPoints that have no subset
		if (!sumVec.is_zero()) {
			centerPoints.row(k) = sumVec;
		}
	}
}

// A function to custom print the class
std::ostream& operator<<(std::ostream& os, const kMeansBase& kmn) {
	os << std::format("We have {} points with {} centers.\n", kmn.numPoints, kmn.kNum);
	os << std::format("The Centroids location found by algorithm is as below:\n", kmn.modelName);
	kmn.centerPoints.print();
	return os;
}

bool kMeansBase::checkErrImprove(const arma::mat prevCenters) {

	arma::mat errMat(prevCenters);
	errMat -= centerPoints;
	errMat = arma::abs(errMat);

	// Check to see all the elements have improved less than a specified amount
	bool checkErr = arma::all(arma::vectorise(errMat) < 0.0001);

	return checkErr;
}