#include <iostream>
#include <fstream>
#include <stdio.h>

using namespace std;

int main ()
{
	std::string name = "John"; int age = 21;
	std::string result;
	std::ofstream myfile;
	char numstr[21]; 
	sprintf(numstr, "%d", age);
	result = name + numstr;
	myfile.open(result.c_str());
	myfile << "HELLO THERE";
	myfile.close();
	return 0;
}
