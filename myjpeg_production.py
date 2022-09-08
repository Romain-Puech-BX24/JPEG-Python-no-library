# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:39:08 2022

@author: Romain Puech

Project : Lossy image compression. Implementing JPEG format in python with no library and the most advanced DCT techniques published.

"""
from math import ceil,floor,sqrt,cos,sin,pi,prod

################################################################
################## Handling PPM formated files #################
################################################################

def remove_comment(line):
    '''
    
    Parameters
    ----------
    line : str
        string to be cleared of comments.

    Returns
    -------
    str
        line without comments.

    '''
    i=line.find("#")
    if i!=-1:
        return line[:i]
    return line

def ppm_tokenize(stream):
    '''
    
    Parameters
    ----------
    stream : TextIOBase
        Already opened ppm file to be tokenized

    Yields 
    ------
    Token: string
         Elements of the file separated by spaces, one by one. Handles comments

    '''
    for line in stream:
        yield from remove_comment(line).strip().split()


####################################


def ppm_load(stream):
    '''
    

    Parameters
    ----------
    stream : TextIOBase
        Already opened ppm file to be loaded

    Returns
    -------
    w : int
        width
    h : int
        height
    img : list
        matrix representing the image loaded from the ppm file

    '''
    tokenized_stream_iterator=ppm_tokenize(stream)
    next(tokenized_stream_iterator)
    w,h,max_depth=int(next(tokenized_stream_iterator)),int(next(tokenized_stream_iterator)),int(next(tokenized_stream_iterator))
    img=[[] for i in range(h)]
    n,row=0,0
    for r in tokenized_stream_iterator:
        try:
            r=int(r)
            g = int(next(tokenized_stream_iterator))
            b = int(next(tokenized_stream_iterator))
        except StopIteration:
            raise TypeError('The file provided is not a valid image file')
        except ValueError:
            #if we can't cast to int r, g or b
            raise TypeError('The file provided is not a valid image file')
        if max(r,g,b)>max_depth or min(r,g,b)<0:
            raise ValueError('One color bit is too high or too low')
            
        try:
            img[row].append((r,g,b))
        except IndexError:
            raise ValueError('invalid w/h combination')
        n+=1
        if n%w==0:
            row+=1
            
    if len(img[-1])!=w:
        raise TypeError('The file provided is not a valid image file')
    return (w,h,img)
 

###########################################

def ppm_save(w,h,img,output):
    '''
    Saves an image in a ppm formated file
    
    Parameters
    ----------
    w : int
        width of the image
    h : int
        height if the image
    img : list
        matrix representing an image
    output : output stream
        output file

    Returns
    -------
    None.
    
    '''
    if w!=len(img) or h!=len(img[0]):
        raise ValueError('Invalid image width or height')
    output.writelines(('P3\n',str(w)+' ',str(h)+'\n','255\n'))
    for i in img:
        for j in i:
            output.write(f'{j[0]} {j[1]} {j[2]}\n')
   
    
##########################################################
###### RGB to YCbCr conversion & channel separation ######
##########################################################


 
def RGB2YCbCr(r, g, b):
    '''
     Takes a pixel’s color in the RGB color space and that converts it in the YCbCr color space
     
    Parameters
    ----------
    r : int
        red channel of the pixel
    g : int
        green channel of the pixel
    b : int
        blue channel of the pixel

    Returns
    -------
    tuple (Y,Cb,Cr) of yellow, blue-difference and red-difference

    '''
    Y =  min(255,max(0,round(0.299*r+0.587*g+0.114*b)))
    Cb = min(255,max(0,round(128-0.168736*r-0.331264*g+0.5*b)))
    Cr = min(255,max(0,round(128+0.5*r-0.418688*g-0.081312*b)))
            
    return(Y,Cb,Cr)

############################################

def YCbCr2RGB(Y, Cb, Cr):
    '''
     Takes a pixel’s color in the YCbCr color space and that converts it in the RGB color space
     
    Parameters
    ----------
    Y : int
        yellow channel of the pixel
    Cb : int
        blue-difference
    Cr : int
        red-difference

    Returns
    -------
    tuple (R,G,B) of red, green and blue components

    '''
    
    R=min(255,max(0,round(Y+1.402*(Cr-128))))
    G=min(255,max(0,round(Y-0.344136*(Cb-128)-0.714136*(Cr-128))))
    B=min(255,max(0,round(Y+1.772*(Cb-128))))
    
    return (R,G,B)

###########################################

def img_RGB2YCbCr(img):
    '''
     takes an image in the RGB-color space and that return a 3-element tuple (Y, Cb, Cr) of matrices

    Parameters
    img: list
        image to be converted

    Returns
    -------
    Y : list
        matrix s.t. Y[i][j] denotes the Y component of img[i][j].
    Cb : TYPE
        matrix s.t. Cb[i][j] denotes the Cb component of img[i][j].
    Cr : TYPE
        matrix s.t. Cr[i][j] denotes the Cr component of img[i][j].
        
    '''
    Y,Cb,Cr=[],[],[]
    for row in img:
        Y.append([])
        Cb.append([])
        Cr.append([])
        for cell in row:
            y,cb,cr = RGB2YCbCr(cell[0], cell[1], cell[2])
            Y[-1].append(y)
            Cb[-1].append(cb)
            Cr[-1].append(cr)
    return Y,Cb,Cr

################################################

def img_YCbCr2RGB(Y, Cb, Cr):
    '''
    

    Parameters
    ----------
    Y : list
        Luminance channel
    Cb : list
        Blue difference
    Cr : list
        Red difference

    Returns
    -------
    img : list
        matrix of tuples

    '''
    img=[]
    for row_Y,row_Cb,row_Cr in zip(Y,Cb,Cr):
        img.append([])
        for cell_Y,cell_Cb,cell_Cr in zip(row_Y,row_Cb,row_Cr):
            R,G,B= YCbCr2RGB(cell_Y,cell_Cb,cell_Cr)
            img[-1].append((R,G,B))
    return img
            
            
        
##########################################################
###################### Subsampling #######################
##########################################################

#attention a et b, a est a height
def subsampling(w,h,C,a,b):
    '''
    split the image into blocks of size a x b that will be encoded by a sample whose value is the average (rounded to the nearest integer) of the block’s samples
    
    Parameters
    ----------
    w : int
        width of the image
    h : int
        height of the image 
    C : list
        matrix to subsample
    a : int
        height of the subsampling
    b : int
        lenght of the subsampling

    Returns
    -------
    img : list
        subsampled image

    '''
    a,b=b,a
    img=[[] for i in range(ceil(h/b))]
    row_index=0
    while row_index*b<=h-1:
        cell_index=0
        while cell_index*a<=w-1:
            ### we start at cell a:b
            s=0
            c=0
            for i in range(a):
                for j in range(b):
                    #we start at index [i][j] and extract a sub-block
                    if row_index*b+j<h and cell_index*a+i<w:
                        s+=C[row_index*b+j][cell_index*a+i]
                        c+=1
            img[row_index].append(round(s/c)) #round it to the nearest int
            cell_index+=1
        row_index+=1
    return img

############################

def extrapolate(w, h, C, a, b):
    '''
    inverse operation of subsampling

    Parameters
    ----------
    w : int
        width of the image after the extrapolation
    h : int
        height of the image after extrapolation
    C : list
        matrix image
    a : int
        height of the image C
    b : int
        width of the image C

    Returns
    -------
    img : list
        matrix image that have been extrapolated

    '''
    a,b=b,a
    img=[[] for i in range(h)]
    for row_index in range(len(C)):
        for cell_index in range(len(C[0])):
            i=0
            while cell_index*a+i<w and i<a:
                j=0
                while row_index*b+j<h and j<b:
                    img[row_index*b+j].append(C[row_index][cell_index])
                    j+=1
                i+=1
                    
    return img  
    


################################################################
####################### Block splitting ########################
################################################################

def block_splitting(w, h, C):
    '''
    takes a channel C and yield all the 8x8 subblocks of the channel, line by line, from left to right.

    Parameters
    ----------
    w : int
        channel's width
    h : int
        channel's height
    C : list
        matrix to be splitted.

    Yields
    ------
    subblock : list
        8*8 sub-block

    '''
    for row_index in range(ceil(h/8)):
        for col_index in range(ceil(w/8)):
            res=[[] for i in range(8)]
            #we start at [i][j] and extract the 8*8 sub-block
            for i in range(8):
                for j in range(8):
                    res[j].append(C[min(h-1,row_index*8+j)][min(col_index*8+i,w-1)])
            yield res
                     
            
################################################################
############## Discrete Cosine Transform (DCT) ################
################################################################

#helper functions#

def delta(i):
    return 1/sqrt(2) if i==0 else 1

#matrix C_n used in the matrix formula of the DCT algorithm
def matrixn(n):
    return [[delta(i)*sqrt(2/n)*cos(pi/n*(j+1/2)*i) for j in range(n)] for i in range(n)]

def vector_times_matrix(vect,matrix):
    return [sum([vect[i]*matrix[i][j] for i in range(len(vect))]) for j in range(len(matrix[0]))]

def matrix_multiplication(matrix1,matrix2):
    return [[sum([matrix1[k][i]*matrix2[i][j]  for i in range(len(matrix1[0]))]) for j in range(len(matrix2[0])) ] for k in range(len(matrix1))]

def transpose(matrix):
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]

#################  
         
def DCT(v):
    '''
    Permorms the discrete cosine transform on a 1D vector
    Parameters
    ----------
    v : list
        vector to be transformed

    Returns
    -------
    list
        transformed vector

    '''
    return vector_times_matrix(v,transpose(matrixn(len(v))))


def IDCT(v):
    '''
    Permorms the inverse discrete cosine transform on a 1D vector
    Parameters
    ----------
    v : list
        vector to be transformed

    Returns
    -------
    list
        transformed matrix

    '''
    return vector_times_matrix(v,matrixn(len(v)))

#####

def DCT2(m, n, A):
    '''
    Permorms the inverse discrete cosine transform on a 2D matrix
    Parameters
    ----------
    v : list
        vector to be transformed
    m: int
        width of A
    n: int
        height of A

    Returns
    -------
    list
        transformed matrix
    '''
    return matrix_multiplication(matrixn(m),matrix_multiplication(A,transpose(matrixn(n))))
    
def IDCT2(m, n, A):
    '''
    Permorms the inverse inverse discrete cosine transform on a 2D matrix
    Parameters
    ----------
    v : list
        vector to be transformed
    m: int
        width of A
    n: int
        height of A

    Returns
    -------
    list
        transformed matrix

    '''
    return matrix_multiplication(matrix_multiplication(transpose(matrixn(m)),A),matrixn(n))

################ DCT_Chen ################

#helper functions

def redalpha(i):
    #we first simplify by 2pi
    i=i%32
    #i = -1 if i>16 else 1
    if i>16:
        s=-1 #negative
    else:
        s=1 #positive
    i=i%16
    if i>8:
        #by trigonometric formulae
        i = 16 - i
        s*=-1
    elif i==8:
        i=0

    return (s,i)
        
def ncoeff8(i,j):#loop less
    if i==0:
        return (1,4)
    alpha_coeff = i*(2*j+1) #the alpha coefficient associated to C_i,j
    s,k = redalpha(alpha_coeff) #its reduced version
    return s,k

'''
M8 = [
    [ncoeff8(i, j) for j in range(8)]
    for i in range(8)
]

def M8_to_str(M8):
    def for1(s, i):
        return f"{'+' if s >= 0 else '-'}{i:d}"

    return "\n".join(
            " ".join(for1(s, i) for (s, i) in row)
            for row in M8
        )
print(M8_to_str(M8))
'''
'''
#observe that cos(x) = cos(-x) so we don't need the sign of ncoeff(i,j)
'''
#matrix C_bar used in the matrix formula of DCT for 8x8 matrices
# /!\ The coefficients are already divided by 2
C_bar = [[ ncoeff8(i,j)[0]*cos(ncoeff8(i,j)[1]*pi/16)/2 for j in range(8)] for i in range(8)]

def omega(i):#helper function
    return (i%2)*(-2)+1 #-1 if i is odd, 1 if it is even


def DCT_Chen(A):
    '''
    Permorms the discrete cosine transform on a 2D vector
    Parameters
    ----------
    A : list
        Matrix to be transformed

    Returns
    -------
    list
        Transformed matrix

    '''
    res=[]
    for k in range(len(A)):
        res.append([])
        v=A[k]#all the components are computed one by one to perform the minimal number of multiplications.
        #Computing v0
        res[-1].append( C_bar[0][0]*sum([v[j] for j in range(8)]) )
        
        #Computing v1, no further optimization possible:
        res[-1].append(sum(  [ (v[j]-v[7-j])*C_bar[1][j] for j in range(4) ]  ))
        
        #computing v2:
        res[-1].append(  C_bar[2][0]*(v[0]+v[7]-v[3]-v[4]) + C_bar[2][1]*(v[1]+v[6]-v[2]-v[5]) )
        
        #computing v3, no further optimization possible:
        res[-1].append(sum(  [ (v[j]-v[7-j])*C_bar[3][j] for j in range(4) ]  ))
        
        #computing v4, same optimization as v0:
        res[-1].append( C_bar[0][0]*(v[0]-v[1]-v[2]+v[3]+v[4]-v[5]-v[6]+v[7]) )
        
        #computing v5, no further optimization possible:
        res[-1].append(sum(  [ (v[j]-v[7-j])*C_bar[5][j] for j in range(4) ]  ))
        
        #computing v6:
        res[-1].append( C_bar[6][0]*(v[0]+v[7]-v[3]-v[4]) + C_bar[6][2]*(-v[1]-v[6]+v[2]+v[5]))
        
        #computing v7, no further optimization possible:
        res[-1].append(sum(  [ (v[j]-v[7-j])*C_bar[7][j] for j in range(4) ]  ))
        
        #22 multiplications : (1+4+2+4+1+4+2+4) = 22
    #The row part is done
    #Let's do the column part i.e, C_bar*A
    final=[[]for i in range(8)]
    for k in range(len(A[0])):#here k is a column index
        v=[res[i][k] for i in range(8)]
        #Computing vk of each row (i.e computing the colum k)
        final[0].append( C_bar[0][0]*sum([v[j] for j in range(8)]) )
        #Computing vk col 2
        final[1].append( sum(  [ (v[j]-v[7-j])*C_bar[1][j] for j in range(4) ]  ) )
        #Computing vk col 3
        final[2].append(   C_bar[2][0]*(v[0]+v[7]-v[3]-v[4]) + C_bar[2][1]*(v[1]+v[6]-v[2]-v[5])  )
        #etc
        final[3].append( sum(  [ (v[j]-v[7-j])*C_bar[3][j] for j in range(4) ]  ) )
        final[4].append(  C_bar[0][0]*(v[0]-v[1]-v[2]+v[3]+v[4]-v[5]-v[6]+v[7])  )
        final[5].append( sum(  [ (v[j]-v[7-j])*C_bar[5][j] for j in range(4) ]  ) )
        final[6].append(  C_bar[6][0]*(v[0]+v[7]-v[3]-v[4]) + C_bar[6][2]*(-v[1]-v[6]+v[2]+v[5]))
        final[7].append( sum(  [ (v[j]-v[7-j])*C_bar[7][j] for j in range(4) ]  ) )

    return final
    
#### Bonus Part: Loeffler algorithm to use only 11 multiplications!

#precomputations
cos3 = cos(3*pi/16)
sin3 =  sin(3*pi/16)
    
cos1 =  cos(pi/16)
sin1 =  sin(pi/16)
    
cos6 =  cos(6*pi/16)
sin6 =  sin(6*pi/16)

def DCT_Loeffler_row(v):
    output=[0]*8
    
    #part 1 of the diagram
    for i in range(4):
        output[i]=v[i]+v[7-i]
    for i in range(4,8):
        output[i] = v[7-i]-v[i]
        
    #part 2 
    output2=[0]*8
    #+/-
    output2[0]=output[0]+output[3]
    output2[1]=output[1]+output[2]
    output2[2]=output[1]-output[2]
    output2[3]=output[0]-output[3]
    #blocks
    output2[4] = output[4]*cos3+output[7]*sin3
    output2[7] = -output[4]*sin3+output[7]*cos3
    output2[5] = output[5]*cos1+output[6]*sin1
    output2[6] = -output[5]*sin1+output[6]*cos1
    
    #part 3
    output3=[0]*8
    
    output3[0] = output2[0]+output2[1]
    output3[1] = -output2[1]+output2[0]
    
    output3[2] =  output2[2]*sqrt(2)*cos6+output2[3]*sqrt(2)*sin6
    output3[3] = -output2[2]*sqrt(2)*sin6+output2[3]*sqrt(2)*cos6
    
    output3[4] =  output2[4]+output2[6]
    output3[5] = -output2[5]+output2[7]
    output3[6] = -output2[6]+output2[4]
    output3[7] =  output2[7]+output2[5]
    
    #part 4
    
    output4 = output3[:4]+[0,0,0,0]
    
    output4[4] = -output3[4]+output3[7]
    output4[5] = sqrt(2)*output3[5]
    output4[6] = sqrt(2)*output3[6]
    output4[7] = output3[4] + output3[7]
    
    return [output4[i]/sqrt(8) for i in (0,7,2,5,1,6,3,4)]

def DCT_Loeffler(A):
    res=[]
    for row in A:
        res.append( DCT_Loeffler_row(row))
    
    res2=[]
    for column_index in range(8):
        column = [res[i][column_index] for i in range(8)] #to have the right order
        res2.append( DCT_Loeffler_row(column))
    
            
    return res2


#helper function
def alpha_by_two(i):
    return cos(i*pi/16)/2

# precompute the matrices
OMEGA = [
    [alpha_by_two(4) for i in range(4)],
    [alpha_by_two(2),alpha_by_two(6),-alpha_by_two(6),-alpha_by_two(2)],
    [alpha_by_two(4),-alpha_by_two(4),-alpha_by_two(4),alpha_by_two(4)],
    [alpha_by_two(6),-alpha_by_two(2),alpha_by_two(2),-alpha_by_two(6)]
    ]
    
THETA = [
       [alpha_by_two(1),alpha_by_two(3),alpha_by_two(5),alpha_by_two(7)],
       [alpha_by_two(3),-alpha_by_two(7),-alpha_by_two(1),-alpha_by_two(5)],
       [alpha_by_two(5),-alpha_by_two(1),alpha_by_two(7),alpha_by_two(3)],
       [alpha_by_two(7),-alpha_by_two(5),alpha_by_two(3),-alpha_by_two(1)]
       ]

NEG_THETA = [[-THETA[i][j] for j in range(4)] for i in range(4)]

#The big matrix. Same as [[*Omega,*Omega],[*Theta,*Neg _Theta]]
OOTNT = [OMEGA[i]*2 for i in range(4)] + [THETA[i]+NEG_THETA[i] for i in range(4)]


#########################################

#cache in order not to do many times the same multiplications. Ensure the lest number of multilications for IDCT_Chen.
def cache_multiplication(matrix_coeff,vhat_coeff,cache):
    sign = 1 if matrix_coeff>=0 else -1 #cache the multiplication up to the sign of the result thus if we computed 4*5, we can have the result for -4*5 as well
    matrix_coeff=abs(matrix_coeff) #this is equivalent to abs(matrix_coeff)
    key=(matrix_coeff,vhat_coeff) #vhat_coeff is always positive
    if key in cache:
        return cache[key]*sign #does not count as a multiplication since sign is +/- 1
    else:
        res=matrix_coeff*vhat_coeff #vhat_coeff is always positive so the sign only depend on the matrix coeff
        cache[key]=res
        return res*sign
        
def IDCT_Chen(A):
    cache=dict()
    '''
    Parameters
    ----------
    A : 8x8 Matrix

    Returns
    -------
    final : 8x8 matrix
        inverse of A by the DCT transform, using less than 32 multiplications per row
    '''
    res=[]
    for vector_index in range(8):
        res.append([])
        vhat_ordered = [A[vector_index][i] for i in (0,2,4,6,1,3,5,7)] #to have the right order
        for k in (0,1,2,3,7,6,5,4):
            res[-1].append(   sum( [ cache_multiplication(OOTNT[i][k],vhat_ordered[i],cache) for i in range(8)] ) )     
    #The row part is done
    final=[[] for i in range(8)] #we initialise all the rows
    #we are going to compute the coefficients column by column
    for column_index in range(8):
        vhat_ordered= [res[i][column_index] for i in (0,2,4,6,1,3,5,7)] #to have the right order
        for k in range(8):#(0,1,2,3,7,6,5,4):
            final[k].append( sum( [ cache_multiplication(OOTNT[i][k],vhat_ordered[i],cache) for i in range(8)] ) ) 

    return final

#########################################
# same without the cache
"""
def IDCT_Chen_vector_omega(vhat):
    vhat_times_alpha = [[0]*8 for i in range(8)]
    res=[0]*8
    
    #We now compute the part multiplied by Omega
    vhat_times_alpha[0][4]= vhat[0]*OMEGA[0][0]
    vhat_times_alpha[2][2]= vhat[2]*OMEGA[1][0]
    vhat_times_alpha[4][4]= vhat[4]*OMEGA[2][0]
    vhat_times_alpha[6][6]= vhat[6]*OMEGA[3][0]
    
    vhat_times_alpha[2][6]= vhat[2]*OMEGA[3][0]
    vhat_times_alpha[6][2]= vhat[6]*OMEGA[1][0]
    
    #No optimization possible for theta excepted the negative sign
    
    #We compute the 'first half' of the result
    res[0] = vhat_times_alpha[0][4]+vhat_times_alpha[2][2]+vhat_times_alpha[4][4]+vhat_times_alpha[6][6] #+theta part
    res[1] = vhat_times_alpha[0][4]+vhat_times_alpha[2][6]-vhat_times_alpha[4][4]-vhat_times_alpha[6][2] #+theta part
    res[2] = vhat_times_alpha[0][4]-vhat_times_alpha[2][6]-vhat_times_alpha[4][4]+vhat_times_alpha[6][2] #+theta part
    res[3] = vhat_times_alpha[0][4]-vhat_times_alpha[2][2]+vhat_times_alpha[4][4]-vhat_times_alpha[6][6] #+theta part
    for i in range(4,8):
        res[i]=res[i-4]
       
    res=[ res[i] for i in (0,1,2,3,7,6,5,4)]
    return res#reordering
    
def IDCT_Chen_vector_theta(vhat):
    vhat_ordered = [vhat[i] for i in (1,3,5,7)]
    res=[]
    for i in range(4):
        res.append(sum( [  THETA[k][i]*vhat_ordered[k] for k in range(4) ] ))
    
    for i in range(4):
        res.append(-res[i])
        
    res=[ res[i] for i in (0,1,2,3,7,6,5,4)]
    return res
    
    
        
            
    
def IDCT_Chen_2(A):
    '''
    Parameters
    ----------
    A : 8x8 Matrix

    Returns
    -------
    final : 8x8 matrix
        inverse of A by the DCT transform, using less than 32 multiplications per row
    '''
    res=[]
    for vector_index in range(8):
        res.append([0]*8)
        omega_part = IDCT_Chen_vector_omega(A[vector_index])
        theta_part = IDCT_Chen_vector_theta(A[vector_index])

        
        for i in range(8):
            res[vector_index][i] = omega_part[i]+theta_part[i]
        
    #The row part is done
    
    final=[]
    for column_index in range(8):
        final.append([0]*8)
        omega_part = IDCT_Chen_vector_omega([ res[i][column_index] for i in range(8) ])
        theta_part = IDCT_Chen_vector_theta([ res[i][column_index] for i in range(8) ])

        
        for i in range(8):
            final[column_index][i] = omega_part[i]+theta_part[i]
            
    final=[ final[i] for i in (0,1,2,3,7,6,5,4)]
    return final
"""

########

def quantization(A,Q):
    '''
     Returns the quantization of A by Q

    Parameters
    ----------
    A : list
        8x8 matrices of numbers (integers and/or floating point numbers)
    Q : list
        8x8 matrices of numbers (integers and/or floating point numbers)

    Returns
    -------
    list
        Quantization of A by Q

    '''
    return [[round(A[j][i]/Q[j][i]) for i in range(8)] for j in range(8)]

def quantizationI(A,Q):
    '''

    Parameters
    ----------
    A : list
        8x8 matrices of numbers (integers and/or floating point numbers)
    Q : list
        8x8 matrices of numbers (integers and/or floating point numbers)

    Returns
    -------
    list
        Inverse quantization of A by Q

    '''
    return [[A[j][i]*Q[j][i] for i in range(8)] for j in range(8)]

LQM = [
  [16, 11, 10, 16,  24,  40,  51,  61],
  [12, 12, 14, 19,  26,  58,  60,  55],
  [14, 13, 16, 24,  40,  57,  69,  56],
  [14, 17, 22, 29,  51,  87,  80,  62],
  [18, 22, 37, 56,  68, 109, 103,  77],
  [24, 35, 55, 64,  81, 104, 113,  92],
  [49, 64, 78, 87, 103, 121, 120, 101],
  [72, 92, 95, 98, 112, 100, 103,  99],
]

CQM = [
  [17, 18, 24, 47, 99, 99, 99, 99],
  [18, 21, 26, 66, 99, 99, 99, 99],
  [24, 26, 56, 99, 99, 99, 99, 99],
  [47, 66, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
]

def S(phi):
    return 200-2*phi if phi>=50 else round(5000/phi)

def Qmatrix(isY, phi):
    '''

    Parameters
    ----------
    isY : boolean
        If isY is True, it returns the standard JPEG quantization matrix for the luminance channel, lifted by the quality factor phi.
         If isY is False, it returns the standard JPEG quantization matrix for the chrominance channel, lifted by the quality factor phi.
    phi : int
        quality vector between 1 and 100

    Returns
    -------
    list
        Standard JPEG quantization matrix for the luminance or chrominance channel

    '''
    if isY:
        return [[ceil((50+S(phi)*LQM[i][j])/100) for j in range(8)] for i in range(8)]
    return [[ceil((50+S(phi)*CQM[i][j])/100) for j in range(8)] for i in range(8)]
    
        

################################################################
######################## Zig-Zag walk ##########################
################################################################

def zigzag(A):
    '''

    Parameters
    ----------
    A : list
        8*8 Matrix to zigzag

    Yields
    ------
    int
        component if the matrix, in a zigzag order

    '''
    row_index=0
    col_index=0
    reverse=False
    yield A[row_index][col_index]
    for nb_diagonal in list(range(1,8))+list(range(6,0,-1)):#the diagonals of the matrix have len 1, then 2,... up to 7 then 6, 5,... down to 1.
        if nb_diagonal%2==1 and not reverse or nb_diagonal%2==0 and reverse:#determining if we have to go up or down. for example, we go up if the len of the diagonal is even and if we are in the first part of the matrix
            col_index+=1
        else:
            row_index+=1
        yield A[row_index][col_index]
        for i in range(nb_diagonal):
            if nb_diagonal%2==1:
                col_index-=1
                row_index+=1
            else:
                col_index+=1
                row_index-=1
            yield A[row_index][col_index]
        if nb_diagonal == 7: 
            reverse=True #after the middle diagonal is done, we have to go down for even len and up for odd len, so it is the opposite as before.
    
    col_index+=1
    yield A[row_index][col_index]
   
################################################################
######################## RLE encoding ##########################
################################################################      
   
def rle0(g):
    '''
    Performs the rle encoding of g.
    
    Parameters
    ----------
    g : generator
        generator that yields integers

    Yields
    ------
    c : int
        number of zeros before i
    i : int
        value of the first non-zero component of g since the last yield.

    '''
    c=0 #nb of 0 counter
    for i in g:
        if i==0:
            c+=1
        else:
            yield (c,i)
            c=0

# :)

        
        
