#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <typeinfo>
#include "functions.h"

using namespace std;

//Scheduler function
int scheduler(int idleIdfs[], int orderlist[])
{
	
}
	
int main()
{	
	//Generating random number of IBF's and products
	int randNumIbfs = (rand() % 25) + 1;
	int randNumProducts = (rand() % 25) + 1;
	
	//Array of Idle IBf's storing the Ibf number which is idle
	int idleIbfs[30];
	int idleIbfCount=0;
	
	//storing the ordered list of products
	int orderlist[30];
	int orderCounts=0;
	
	vector <Ibf> Ibfs;
	vector <Product> products;
	cout << "Creating IBFs - " << randNumIbfs << endl;
	//creating IBF's	
	for(int i=0; i< randNumIbfs; i++){Ibfs.push_back(Ibf(i, "idle", "hot"));}

	cout << "Creating Products - " << randNumProducts << endl;
	//creating products	
	for(int i=0; i< randNumProducts; i++){products.push_back(Product("A", i, "ordered"));}


	for(int j=0; j<randNumIbfs; j++)
		cout << " IBF number " << Ibfs[j].getIbfNumber() << " is in " << Ibfs[j].getOperationalStatus() << "state" << endl;	

	for(int k=0; k<randNumProducts; k++)
		products[k].toStringProduct(k);
		
	for(int l=0;l<randNumIbfs; l++){	
		
		if(Ibfs[l].getOperationalStatus() == "idle")
			idleIbfs[idleIbfCount]=Ibfs[l].getIbfNumber();
			idleIbfCount++;
	}
	
	//transport function call to add the product number 4 to IBF4	
	transport(products[4] ,Ibfs[4]);
	//transport function call to add the product number 5 to IBF4
	//nothing happens, as status of IBF 4 is busy	
	transport(products[5] ,Ibfs[4]);
	
	return 0;
}
