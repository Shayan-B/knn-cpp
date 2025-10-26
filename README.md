# KMeans-CPP

This repository implements the K-Means and K-Means++ clustering algorithms in C++ using Visual Studio 2022. The project leverages the Armadillo library for efficient linear algebra operations, such as matrix manipulations and distance calculations. The folder structure and build configuration are tailored for Visual Studio 2022, including solution (.sln) and project (.vcxproj) files.

## Features

- Standard K-Means algorithm for unsupervised clustering.
- K-Means++ initialization method for improved centroid selection and convergence.
- Utilizes Armadillo for high-performance matrix operations.
- Example code demonstrating usage on sample datasets.
- Configured for easy building and debugging in Visual Studio 2022.

## Prerequisites

- **Visual Studio 2022**: Required for building and running the project. Ensure you have the C++ desktop development workload installed.
- **Armadillo Library**: Included in the project (headers and pre-built libraries are vendored in the `lib/armadillo` directory for convenience). No separate installation is needed, as the project files are pre-configured to link against it.

## Folder Structure

The repository follows a standard Visual Studio 2022 project layout:

- **`KMeans-CPP.sln`**: The solution file to open in Visual Studio.
- **`KMeans-CPP/`**: The main project directory.
  - **`KMeans-CPP.vcxproj`**: The project file.
  - **`source/`**: Contains the C++ source files (e.g., `main.cpp`, `kmeans.cpp`, `kmeans_plus_plus.cpp`).
  - **`include/`**: Header files for the algorithms.
  - **`lib/armadillo/`**: Vendored Armadillo library (headers in `include/`, libs in `lib/`).
  - **`data/`**: Sample datasets for testing (e.g., CSV files with point coordinates).
  - **`build/`**: Output directory for compiled binaries (Debug/Release configurations).
- **`README.md`**: This file.
- **`LICENSE`**: License information (e.g., MIT).

## Building the Project

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/KMeans-CPP.git
   cd KMeans-CPP
   ```

2. Open the solution in Visual Studio 2022:
   - Double-click `KMeans-CPP.sln` or open it via File > Open > Project/Solution.

3. Configure the build:
   - Set the active configuration (Debug/Release) and platform (x64 recommended for Armadillo compatibility).
   - The project is pre-configured to include Armadillo paths in the include directories and linker settings.

4. Build the solution:
   - Press `Ctrl + Shift + B` or right-click the project in Solution Explorer and select Build.
   - The executable will be generated in the `build/Debug/` or `build/Release/` folder (e.g., `KMeans-CPP.exe`).

If you encounter linker errors related to Armadillo, ensure the library paths are correctly set in Project Properties > VC++ Directories and Linker > Input.

## Usage

The project builds a console application that demonstrates the K-Means and K-Means++ algorithms on sample data.

### Running the Executable

1. After building, run the executable from the command line:
   ```
   .\build\Debug\KMeans-CPP.exe
   ```

2. Command-line arguments (optional):
   - `--k <int>`: Number of clusters (default: 3).
   - `--input <file>`: Path to input CSV file (format: rows as points, columns as features; default: `data/sample.csv`).
   - `--method <string>`: "kmeans" or "kmeans++" (default: "kmeans++").
   - `--iterations <int>`: Maximum iterations (default: 100).

Example:
```
.\build\Debug\KMeans-CPP.exe --k 4 --input data/iris.csv --method kmeans++
```

The program will output cluster assignments, centroids, and basic metrics (e.g., inertia).

### Integrating as a Library

To use the algorithms in your own C++ project:
- Include the headers: `#include "include/kmeans.h"`
- Link against the built library or copy the source files.
- Example usage:
  ```cpp
  #include <armadillo>
  #include "kmeans.h"

  int main() {
      arma::mat data = arma::randu<arma::mat>(100, 2);  // Sample 100 points in 2D
      int k = 3;
      arma::mat centroids;
      arma::uvec labels = kmeans_plus_plus(data, k, centroids, 100);
      // Process labels and centroids...
      return 0;
  }
  ```

## Dependencies

- **Armadillo**: Version X.Y.Z (included in the repository). For more details, visit [Armadillo's official site](http://arma.sourceforge.net/).
- No external dependencies beyond Visual Studio 2022.

## Contributing

Contributions are welcome! Feel free to open issues for bugs or feature requests. To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -am 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by standard machine learning clustering techniques.
- Thanks to the Armadillo team for their excellent linear algebra library.
