#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <typeinfo>

using namespace std;

//creating IBF class with attributes - number, Ibf Status and Burn Service
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

string Ibf:: operationStatus = "idle";
//contructor is called every single time, animal object is created.
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

//creating class product to manage products
//creating product class with attributes - name, Prodcut ID and Product Status
class Product{
	private:
		string productName;
		int productId;
		string productStatus;
	
	public:
		string getProductName(){return productName;}
		string getProductStatus(){return productStatus;}
		int getProductId(){return productId;}
		void setProductStatus(string status){productStatus = status;}
	
	Product(string, int, string);
	Product();
	
	void toStringProduct(int number);
	
};

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

//Adding product in IBF
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
