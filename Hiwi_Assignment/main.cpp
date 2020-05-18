
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using std::cout;
using std::endl;

using PixelCoordinates = std::pair<int, int>;
using LUT = std::vector<PixelCoordinates>;

//======================================================================================================================
//                                             GLOBAL VARIABLES
//======================================================================================================================

constexpr auto KInputDir{"/data/workspace/cameraCalibration/surroundView/lutDemo/"};
constexpr int KViewID{14};

//======================================================================================================================
//                                          FUNCTIONS DECLARATION
//======================================================================================================================

template<typename T>
void readPOD(T& data, std::istream* stream);
std::string buildLUTFilename(std::string const& inputDir, int viewID);
std::string buildImageFilename(std::string const& inputDir, int viewID);
LUT loadLUT(std::string const& filename);
void run();

//======================================================================================================================
//                                          FUNCTIONS IMPLEMENTATION
//======================================================================================================================

template<typename T>
inline void readPOD(T& data, std::istream* stream)
{
    stream->read(reinterpret_cast<char*>(&data), sizeof(T));
}

//----------------------------------------------------------------------------------------------------------------------

inline std::string buildLUTFilename(std::string const& inputDir, int viewID)
{	
	std::string LUT_filename;
	LUT_filename = *inputDir + "lut_" + to_string(viewID) + ".bin";
	cout << LUT_filename;
    // TASK: Build the LUT filename. For instance, for the given GLOBAL VARIABLES above, it should return
    // "/data/workspace/cameraCalibration/surroundView/lutDemo/lut_14.bin"
    return {};
}

//----------------------------------------------------------------------------------------------------------------------

inline std::string buildImageFilename(std::string const& inputDir, int viewID)
{

	std::string imageFilename;
	imageFilename = *inputDir + "images/000000" + to_string(viewID) + ".png";
	cout << image_Filename;	
    // TASK: Buid the input image filename. For instance, for the given GLOBAL VARIABLES above, it should return
    // "/data/workspace/cameraCalibration/surroundView/lutDemo/images/00000014.png"
    return {};
}

//----------------------------------------------------------------------------------------------------------------------

LUT loadLUT(std::string const& filename)
{
    std::ifstream inFile{filename.c_str(), std::ios_base::binary};
    if (!inFile.good())
    {
        cout << "\nFailed to open file " << filename << endl;
        return {};
    }

    std::size_t size;
    inFile >> size;
    inFile.seekg(1, std::ios::cur);     // skip new line character

    LUT lut(size);
    PixelCoordinates::first_type x;
    PixelCoordinates::second_type y;
    for (auto& elem : lut)
    {
        readPOD(x, &inFile);
        readPOD(y, &inFile);
        elem = {x, y};
    }
    return lut;
}

//----------------------------------------------------------------------------------------------------------------------

void run()
{
    cout << "\nLoading LUT... " << std::flush;
    const auto KLUT{loadLUT(buildLUTFilename(KInputDir, KViewID))};
    if (KLUT.empty())
    {
        cout << "Aborting." << endl;
        return;
    }
    cout << "done.";

    cout << "\nLoading input image... " << std::flush;
    // TASK: Load input image here
    cout << "done.";

    cout << "\nBuilding top-down view... " << std::flush;
    // TASK: Create the top-down view here
    cout << "done." << endl;

    // TASK: Display the original fisheye and the top-down view. Wait for a key to be pressed

    cout << "\n============================================================";
    cout << "\n                      Time Summary";
    cout << "\n============================================================";
    // TASK: Print the following:
    //       - total execution time
    //       - time for loading the LUT
    //       - time for loading the input image
    //       - time for the creation of the top-down image
}

//======================================================================================================================
//                                                  MAIN
//======================================================================================================================

int main(int argc, char* argv[])
{
    run();
    return EXIT_SUCCESS;
}
