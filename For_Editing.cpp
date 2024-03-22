double inLmin = (i == 0)? LColRecv_in[j] : in[IDX(i-1,j)];
double inRmax = (i == endx-1 && G_ex != G_Nx)? RColRecv_in[j] : in[IDX(i+1,j)];
// cout << i<< "   " << j << endl;
double inBmin = (j == 0)? BRowRecv_in[i] : in[IDX(i,j-1)] ;
double inTmax = (j == endy-1 && G_ey != G_Ny)? TRowRecv_in[i] : in[IDX(i,j+1)] ;


int jm1 = 0, jp1 = 2;

    for (int j = By; j < Ey; ++j) {
        for (int i = Bx; i < Ex; ++i) {


            out[IDX(i,j)] = ( -     inLmin
                              + 2.0*in[IDX(i,   j)]
                              -     inRmax)*dx2i
                          + ( -     inBmin
                              + 2.0*in[IDX(i,   j)]
                              -     inTmax)*dy2i;
        }
        jm1++;
        jp1++;
    }