all: program

program: main.o Ibf.o Product.o Transport.o
	g++ Ibf.o Product.o Transport.o -o program

main: main.cpp
	g++ -c main.cpp

Transport.o: Transport.cpp
	g++ -c Transport.cpp 

Product.o: Product.cpp
	g++ -c Product.cpp 

Ibf.o: Ibf.cpp
	g++ -c Ibf.cpp 

clean:
	rm -rf *o hello