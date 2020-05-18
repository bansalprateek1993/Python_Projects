#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <typeinfo>
#include "functions.h"

using namespace std;

string Ibf:: operationStatus = "idle";
//contructor is called every single time, Ibf object is created.
Ibf::Ibf(int number, string operationStatus, string burnService){
	
	this -> number = number;
	this -> operationStatus = operationStatus;
	this -> burnService = burnService;

//	Ibf::numOfIbfs++;
}

Ibf::Ibf(){
//	IBF:numOfIbfs++;
}

void Ibf::toString(int number){
	cout << "IBF with number" << number << " is " << this->operationStatus << " and running " << this->burnService << endl;
}
