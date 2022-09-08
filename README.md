# JPEG-Python-no-library
A program implementing the JPEG format in Python, using the most advanced compression techniques published and no library.

From PPM ASCII to JPEG, it performs: 
-PPM reading and tokenization
-RGB to YCbCr conversion & channel separation
-Subsampling on the Cb and Cr channels
-Block-splitting in a 8x8 matrix 
-2D DCT-II
-Matrix quantization (different for the chrominance and luminance channels as the human eye is more sensitive to luminance)
-Rle encoding using a zigzag walk
-entropy encoding and saving (yet to be implemented).

The DCT have been implemented with 3 different algorithms: 
-the basic one
-Chen's algorithm, that computes the 2D DCT-II transform of 8x8 matrix using only 352 multiplications (where the naive algorithm uses 1024 multiplications)
-Loeffler algorithm, as described in this paper: to reduce the number of multiplications to 176. 

The inverse of all the transformations is also implemented. After the DCT, the matrix is encoded using rle-encoding with a zigzag walk to maximize the number of consecutive zeros. 
The very last step, the entropy encoding and saving of the final file, is yet to be implemented.
