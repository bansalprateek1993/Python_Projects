#include <iostream>
#include <string>
#include <stdlib.h>

using namespace std;

int main()
{
	int intnumberguessed = 1;
	string numberguessed;
	
	do{
		cout << "guess any number btw 0 to 10";
		getline(cin, numberguessed);
//		intnumberguessed = stoi(numberguessed);
		cout << numberguessed;
		cout << intnumberguessed << "\n";
		
	}while(intnumberguessed !=4 );
	
	return 0;
}
