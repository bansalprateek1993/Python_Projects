#include <iostream>
#include <string>
#include <stdlib.h>

using namespace std;
int main()
{
	int randNum = (rand() % 26) + 1;
	while(randNum != 25)
	{
		cout << "Number is " << randNum << endl;
		randNum = (rand() % 26) + 1;
	}
	return 0;
}
