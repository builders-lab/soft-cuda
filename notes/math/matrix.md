## Matrix Multiplication

For performance-critical paths, transpose B before calling `tensor_mul`. 
The library provides `tensor_transpose(B)` which returns a new contiguous tensor.
Autograd tracks both ops correctly.

In situation that `tensor_transpose(B)` was not called before `tensor_mul` the operation is dispatched to `tensor_mul_naive` implicityly.

For transpose operation `tensor_transpose` performs the deep 
tranposed copy of the original matrix.
