# training.mod

#### sets and parameters ####
# variable ranges
param x1_U;
param x1_L;
param x2_U;
param x2_L;

# sample indices
param n1;
param n2;
set S_x1 := 1..n1;
set S_x2 := 1..n2;

# variable values
param x1{i in S_x1};
param x2{j in S_x2};

# function values
param f{i in S_x1, j in S_x2};

# training set (input indices)
set T := {S_x1,S_x2};

## neural network graph
param nI;
param nH;
param nO;

set I := 1 .. nI; 		# input nodes
set H := nI+1 .. nI+nH; # hidden nodes
set O := nH+1 .. nH+nO; # output nodes
set V := I union H union O;
set AIH := {I,H};
set AHO := {H,O};
set AIO := {I,O};
set A := AIH union AHO union AIO;
#set A; # adjacency matrix

# in-neighbourhoods
set N{i in V} := {j in V : (j,i) in A};


#### decision variables ####
var v{T,V};
var w{A};
var b{V diff I};


#### objective function ####
minimize nnerr:
	sum{(i_x1,i_x2) in T, o in O} (v[i_x1,i_x2,o]-f[i_x1,i_x2])^2;


#### constraints ####
# input
subject to input_x1{(i_x1,i_x2) in T}:
	v[i_x1,i_x2,1] = x1[i_x1];
subject to input_x2{(i_x1,i_x2) in T}:
	v[i_x1,i_x2,2] = x2[i_x2];

# hidden nodes with ReLU
subject to hidden{(i_x1,i_x2) in T, h in H}:
	v[i_x1,i_x2,h] = max(0,sum{i in N[h]} w[i,h]*v[i_x1,i_x2,i] + b[h]);

# output node is linear
subject to output{(i_x1,i_x2) in T, o in O}:
	sum{h in N[o]} w[h,o]*v[i_x1,i_x2,h] + b[o] = v[i_x1,i_x2,o];