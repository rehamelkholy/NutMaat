#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
void getspectrum();

int main()
{
  char inspec[80],outspec[80],buffer[80],buf2[100];
  FILE *in,*out,*lis;
  int errstat;
  float x[1800],y[1800],max;
  long i,k;

  lis = fopen("list.txt","r");

  while(fscanf(lis,"%s",buffer) != EOF) {
    strcpy(inspec,"/usr/local/mkclass/libr18/");
    strcat(inspec,buffer);
    strcpy(outspec,buffer);
    sprintf(buf2,"srebin0 %s tmp1.rbn05 3800.0 4620.0 0.05",inspec);
    errstat = system(buf2);
    sprintf(buf2,"smooth2 tmp1.rbn05 tmp.spc 0.05 2.25 0.5");
    errstat = system(buf2);
    getspectrum("tmp.spc",x,y,&k);
    max = 0.0;
    for(i=0;i<k;i++) {
      if(x[i] > 4100.0 && x[i] < 4340.0) {
        if(y[i] > max) max = y[i];
      }
    }
    for(i=0;i<k;i++) y[i] /= max;
    out = fopen(outspec,"w");
    for(i=0;i<k;i++) fprintf(out,"%f %f\n",x[i],y[i]);
    fclose(out);
  }
  fclose(lis);
  return(0);
}
