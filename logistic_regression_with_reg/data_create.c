
#include <stdio.h>

int main(){

FILE *fp1;
fp1 = fopen("input", "w");
for (int i=0; i<50; i++){
        fprintf(fp1, "%lf  \t %lf \t %d \n", (i*0.1), (i+0.1), 1);
}

for (int i=0; i<50; i++){
        fprintf(fp1, "%lf  \t %lf \t %d \n", (-i*0.1), -(i+0.1), 0);
}
}
