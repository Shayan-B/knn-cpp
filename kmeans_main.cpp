#include <armadillo>
#include "include/kMeans.h"
//#include "include/kmeanspp.h"

import <cmath>;
import <string>;
import <iostream>;

using std::string;

int main() {
    kMeans kmeansPts(3, 100);
    kmeansPts.searchCenters();
    
    return 0;
}