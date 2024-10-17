#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

void getspectrum(infile,x,y,k)
     char infile[];
     float x[],y[];
     long *k;
{

  long l;
  char buffer[200],*tmp;
  FILE *in;

  in = fopen(infile,"r");
  l = 0;
  while(fgets(buffer,190,in) != NULL) {
    if(buffer[0] == '#') continue;
    tmp = strtok(buffer," ");
    x[l] = atof(tmp);
    tmp = strtok(NULL," ");
    y[l] = atof(tmp);
    l++;
  }
  *k = l;
  fclose(in);
  return;
}
