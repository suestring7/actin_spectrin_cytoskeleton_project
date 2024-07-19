import cython

def grid2D(int[:,:] grid, int[:] xc, int[:] yc, int x_size, int y_size, int n):
    cdef int x
    cdef int y
    cdef int i
    for i in range(n):
        x=xc[i]
        y=yc[i]
        if( x>0  and x<x_size and y>0 and y<y_size):
            grid[x,y]+=1
    return grid