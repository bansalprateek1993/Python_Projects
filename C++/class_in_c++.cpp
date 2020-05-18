//Full tutorial available at http://www.newthinktank.com/2014/11/c-programming-tutorial/
#include <iostream>
#include <string>

using namespace std;

class Animal
{
// Each class has attrbutes and capabilites:
// Attributes are height, wieght ------------------- VARIABLES
// Capabilites are run, eat ------------------------ FUNCTION

//Private is only for class, that is only function present inside class can change the values.
	private:
		int height;
		int weight;
		string name;
		
		//static means it is variable value is shared by all every object of type animal
		static int numOfAnimals; 
		
	//Have public methods so that value inside private method of the same class is accessible.
	public:
		int getHeight(){ return height; }
		int getWeight(){ return weight; }
		string getName(){return name; }
		
		void setHeight(int cm){height = cm;}
		void setWeight(int kg){weight = kg;}
		void setName(string animalName){name = animalName;}
		
		void setAll(int, int , string);
		
		//Creating constructor
		// name should be same as of class.
		//Will be created whenever the object is created 
		// Pass the heigh, weigh and name
		// constructur is going to handle the creation of every object
		Animal(int, int, string);
		
		//creating destructor
		~Animal();
		
		//One more constructor, which doesnt recieve anything.
		Animal();
		
		//static method is attached to class but not the objects.
		// Can only access static member variable
		static int getNumOfAnimals(){ return numOfAnimals; }
		
		void toString();
};

//After creating class, now we need to declare evrything

//First static variable
int Animal::numOfAnimals = 0;

void Animal::setAll(int height, int weight, string name){
	//obejct specific height, not just generic height
	// whenever class is created there are no objects created
	// 	we want to revert to specific animal height
	
	this -> height = height;
	this -> weight = weight;
	this -> name = name;
	Animal::numOfAnimals++;
}

//contructor is called every single time, animal object is created.
Animal::Animal(int height, int weight, string name){
	
	this -> height = height;
	this -> weight = weight;
	this -> name = name;
	Animal::numOfAnimals++;
}

//Deconstructor
Animal::~Animal(){
	cout << "Animal" << this->name<< "Destroyed" << endl;
}


//overloaded constructor, whenever no values are passed
Animal::Animal(){
	//As we have created another animal
	Animal::numOfAnimals++;	
}

void Animal::toString(){
	cout << this -> name << "is" << this->height << "cms tall and" << this->weight << "Kgs in weight" <<endl;
}



//Now we can use all the methods and attributes from Animal class to our Dog class
class Dog : public Animal{

private:
	string sound = "Woaf";
public:
	void getSound() {
		cout << sound << endl;
	}	
	
	//New constructor 
	Dog(int, int, string, string);
	Dog() :Animal(){};
	
	void toString();
	
};

//Now we need to define everything for dog class that we are going to change.

//using animal constructor in Dog constructor, As attributes are shared.
Dog::Dog(int height, int weight, string name, string bark):
	Animal(height, weight,name){
		
		this -> sound = bark;
	}

void Dog::toString(){
	
	//As varibles are private need to use get method to access
	cout << this -> getName() << " is " << this -> getHeight() <<
	"cms tall and " << this -> getWeight() << "kgs in weight and says " <<
	this -> sound;
}

//polymorphism
class Animal{
	public:
		void getFamily(){cout << "We are Animals" <<endl;}
		
		//When we know that animal(base class) is a method overwritten by a subclass. 
		virtual void getClass(){
			cout << "I am an Animal" << endl;}
};

class Dog : public Animal{
	
	public:
		getClass() { cout << "I am a Dog" << endl;
		}
};

int main()
{
	//using normal constructor
	Animal fred;
	fred.setHeight(33);
	fred.setWeight(10);
	fred.setName("fred");
	
	cout << fred.getName() << "is" << fred.getHeight() << 
	"cms tall and" << fred.getWeight() << "Kgs in weight" <<endl;
	
	//using constructor with attributes
	Animal tom(36,15,"Tom");
	
	cout << tom.getName() << "is" << tom.getHeight() << 
	"cms tall and" << tom.getWeight() << "Kgs in weight" <<endl;
	
	Dog spot(38, 16, "Spot", "Woof");
	
	cout << "Number of Animals" << Animal::getNumOfAnimals() << endl;
	spot.getSound();
	tom.toString();
	spot.toString();
	
	spot.Animal::toString();
	
	
	Animal *animal = new Animal;
	Dog *dog = new Dog;
	
	animal -> getClass();
	dog->getClass();
	return 0;
	
}
