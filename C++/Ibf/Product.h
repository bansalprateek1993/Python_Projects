#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <typeinfo>

using namespace std;
//creating class product to manage products
//creating product class with attributes - name, Prodcut ID and Product Status
class Product{
	private:
		std::string productName;
		int productId;
		std::string productStatus;
	
	public:
		std::string getProductName(){return productName;}
		std::string getProductStatus(){return productStatus;}
		int getProductId(){return productId;}
		void setProductStatus(string status){productStatus = status;}
	
	Product(string, int, string);
	Product();
	
	void toStringProduct(int number);
	
};
