# JPEG-Python-no-library
A program implementing the JPEG format in Python, using the most advanced DCT algorithm published, and no external library.

From PPM ASCII format to JPEG, it performs:
-PPM reading and tokenization
-RGB to YCbCr conversion & channel separation (using the JPEG standard that uses a modified version of the ITU-R BT.601 standard)
-Subsampling on the lumiance channels, since the human eye is less sentitive to luminance information than to chrominance's.)
-Block-splitting in 8x8 matrices
-2D discrete cosine transform (DCT) of the matrices using Chen's or Loeffler's algorithms (most effective algorithms for 2D DCT-II).
-Matrix quantization (different for the chrominance and luminance channels as the human eye is more sensitive to luminance)
-Rle encoding, using a zigzag walk (to maximize the number of consecutive zeros, using observations of the repartition of the zeros in the matrices after the DCT).
-entropy encoding and saving (yet to be implemented).

The DCT have been implemented with 3 different algorithms: 
-the basic one
-Chen's algorithm, that computes the 2D DCT-II transform of 8x8 matrix using only 352 multiplications (where the naive algorithm uses 1024 multiplications)
-Loeffler algorithm, as described in this paper: https://hal.archives-ouvertes.fr/hal-01797957/file/FINAL.pdf to reduce the number of multiplications to 176. 

The inverse of all the operations is also implemented (excpet the inverse DCT with Loeffler's algorithm).
The very last step is yet to be implemented.
