#include <iostream>
#include <math.h>
#include <time.h>
using namespace std;

int main(){
  int a = 100, b = 100,i,j,k=0,N;
  double diff = 1, old, precise = 0.0000001;
  double result[a][b] = {0.0};
  clock_t start, end;
  double total_time;
  /*produce the a,b the number of the row and column of mesh
  i,j,k for the loop, N is for the input, the number of inerations
  diff, the difference and old to reserve the old to put the old number
  in it, precise means the how small the difference need to be,
  result be the matrix representing the mesh, and start, end, toatl_time
  to calculate the time*/
  start = clock();
  for(i=0;i<a;i++){
    result[i][b-1] = 1.0;
  }
  /*produce the original matrix, oneside is 1 the other is 0*/
  cout<<"Please input the number of iterations:"<<endl;
  cin>>N;
  cout<<"what you input is:"<<N<<endl;
  /*input the number of interations*/
  if(N > 0){
    for(k=0;k<N;k++){
      diff = 0.0;
      for(i=1;i<a-1;i++){
	for(j=1;j<b-1;j++){
	  old = result[i][j];
	  result[i][j] = (result[i-1][j]+result[i+1][j]+
			  result[i][j-1]+result[i][j+1])/4;	  
	  diff = pow((pow(diff,2)+pow((result[i][j]-old),2)),1.0/2);
	}
      }
    }
    cout<<"The difference is:"<<diff<<endl;
    /*if N is a positice integer, we think N is the most times of interations
    and we use gaussseidel method to calculate the PDE, because I found if 
    we input a string N will be a 0, so we use N > 0 as a condition*/
  }else{
    while(diff>=precise){
      diff = 0.0;
      k = k + 1;
      for(i=1;i<a-1;i++){
	for(j=1;j<b-1;j++){
	  old = result[i][j];
	  result[i][j] = (result[i-1][j]+result[i+1][j]+
			  result[i][j-1]+result[i][j+1])/4;	  
	  diff = pow((pow(diff,2)+pow((result[i][j]-old),2)),1.0/2);
	}
      }
    }
    cout<<"the level of accuracy is "<<precise<<endl;
    cout<<"the difference is "<<diff<<endl;
    cout<<"the number of iterations is "<<k<<endl;
    /*if N is not a positice integer, we use precise to test the convergence of 
    interations and we use jacobi method to calculate the PDE*/
  }
  end = clock();
  total_time = (double)(end - start)/CLOCKS_PER_SEC;
  /*calculate the time of execution*/
  cout<<"Total time is "<<total_time<<" seconds"<<endl;
  return 0;
}
