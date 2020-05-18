#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <typeinfo>
//creating IBF class with attributes - number, Ibf Status and Burn Service

using namespace std;

class Ibf{
	private:
		int number;
		static string operationStatus;
		string burnService;
		
//		static int numOfIbfs;
		
	public:
		static string getOperationalStatus(){ return operationStatus;}
		string getBurnService(){return burnService;}
		int getIbfNumber(){return number;}
		static void setIbfStatus(string ibfStatus){operationStatus = ibfStatus;}
		
		//declaring constructor to initialize the IBF's
		Ibf(int, string, string);
		
		//overloaded constructor, if we want to add the details later
		Ibf();
		
		
//		static int getNumOfIbf(){ return numOfIbfs;}
		
		void toString(int number);
	
};


