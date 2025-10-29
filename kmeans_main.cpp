#include <armadillo>
#include "include/kMeans.h"
import <iostream>;

int main() {
    kMeans kmeansPts(3, 100);
    kmeansPts.searchCenters();

    std::cout << "Programs is done.\n";
    std::cout << kmeansPts;
    
    return 0;
}