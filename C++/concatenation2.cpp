#include <iostream>
#include <fstream>
#include <stdio.h>

using namespace std;

constexpr auto KInputDir{"Prateek"};
constexpr int KViewID{14};

const char *one = "Hello ";
int main ()
{	
	std::string result;
	std::ofstream myfile;
	result = std::string(KInputDir) + "000000" +std::to_string(KViewID);
	myfile.open(result.c_str());
	myfile.close();
	return 0;
}
