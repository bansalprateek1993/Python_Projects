#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <typeinfo>
#include "functions.h"
//Adding product in IBF
using namespace std;

int add(Product productInfo,Ibf ibfInfo)
{	
	productInfo.setProductStatus("Processing");
	ibfInfo.setIbfStatus("Busy");	
		
	cout << "--------------" << endl;
	ibfInfo.toString(ibfInfo.getIbfNumber());
	cout << "--------------" << endl;
	return 0;
}	

//removing product from IBF
int remove(Ibf ibfInfo){
	ibfInfo.setIbfStatus("idle");				
	return 0;
}

//Transport System to add and remove the product from/to IBFs
void transport(Product productInfo, Ibf ibfInfo){
	if(productInfo.getProductStatus()  == "ordered" && ibfInfo.getOperationalStatus() == "idle"){
		add(productInfo, ibfInfo);
		cout << "Product is added to IBF number - " << ibfInfo.getIbfNumber()<< endl;
	}
	else if(productInfo.getProductStatus()  == "Processed"){
		remove(ibfInfo);	
		cout << "Product is removed from IBF number - " << ibfInfo.getIbfNumber() << endl;
	}
}
