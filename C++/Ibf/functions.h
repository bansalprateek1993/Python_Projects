#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <typeinfo>
#include "Product.h"
#include "ibf.h"

using namespace std;

void toStringProduct(int Id);
void transport(Product productInfo, Ibf ibfInfo);
int add(Product productInfo,Ibf ibfInfo);
int remove(Ibf ibfInfo);
void toString(int number);
