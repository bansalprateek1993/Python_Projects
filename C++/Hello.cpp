#include<iostream>

using namespace std;
int main()
{	
	cout << "Hello World" << endl;
	
	// constant value cant be changed and Write it in upper case
	const double PI = 3.145124124143124;
	
	// character sourounded by single character - takes 1 byte inside memory
	char grade = 'A';
	int c = 2147483647;
	
	//float is used till 6 digit places
	// doubles kind of float but accurate arround 15 values 
	
	cout << "Grade" << grade;
	cout << "Largest integer" << c;
	//short int : at least 16 bits
	// long int : at least 32 bits
	// long long int : atleast 64 bits
	// unsigned int : same size as signed version
	// long double : not less then double
	
	// To find the size
	cout << "Size of variable\n" << sizeof(grade);
	return 0;
}
