#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <typeinfo>
#include "functions.h"

using namespace std;

Product::Product(string productName, int productId, string productStatus){
	this -> productId = productId;
	this -> productStatus = productStatus;
	this -> productName = productName;
}

Product::Product(){}

//toString Product - to get the knowledge of the Product in one line.
void Product::toStringProduct(int Id){
	cout << "product with ID " << Id << " has status " << this->productStatus << " and name is " << this->productName << endl;
}


