# Data Pipe
Pipeline API to manipulate dataframes.

Data Pipe is a framework that wraps Pandas Data Frames to provide a more fluid method to manipulate data. 

Basic concepts:
- Every operation is performed in place. The Data Pipe object keeps one and only one reference to a pandas Data Frame that is constantly updated. 
- ‎Every operation returns a reference to self, which allows chaining methods fluidly. 
- Every method called is recorded internally to provide improved reproducibility and understanding of the preparation pipeline. The exception is the "load" method.
- ‎Data Pipe calls of unimplemented methods default to the internal Data Frame object. This allows quickly accessing some methods, such as shape and head, but please be aware that those calls are not recorded and do not return Data Pipe objects. If it's necessary to use an unimplemented function, please use the Update method to keep manipulating the Data Pipe. 


