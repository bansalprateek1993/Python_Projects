#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <cstdint>
#include <windows.h>
using namespace std;

#define NUM_THREADS 4

void *PrintHello(void *threadid) {
   int tid;
   tid = (intptr_t)threadid;
   cout << "Hello World! Thread ID, " << tid << endl;
//   sleep(5);
   pthread_exit(NULL);
}

void *PrintHello1(void *threadid) {
   int tid;
   tid = (intptr_t)threadid;
   cout << "Hello Worlddddddddd! Thread ID \n" << tid << endl;
   Sleep(2000);
   pthread_exit(NULL);
}
int main () {
   pthread_t threads[NUM_THREADS];
   int rc;
   int i;
   
   rc = pthread_create(&threads[i], NULL, PrintHello1, (void *)1);
   for( i = 1; i < NUM_THREADS; i++ ) {
      cout << "main() : creating thread, " << i << endl;
      rc = pthread_create(&threads[i], NULL, PrintHello, (void *)i);
      
      if (rc) {
         cout << "Error:unable to create thread," << rc << endl;
         exit(-1);
      }
   }
   pthread_exit(NULL);
}
