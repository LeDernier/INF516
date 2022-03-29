# training_testing.mod

#### sets and parameters ####

# regularization
param lambda;

# variable ranges
param x1_U;
param x1_L;
param x2_U;
param x2_L;

# sample indices
param n1;					# number of different values in dimension 1 (whole set)
param n2;					# number of different values in dimension 2 (whole set)
param nt1;   				# number of different values in dimension 1 (training set)
param nt2;   				# number of different values in dimension 2 (training set)
set T_1 := 1..nt1;          # indices of the 1st coordinate of the training set 
set T_2 := 1..nt2;			# indices of the 2nd coordinate of the training set
set Tv_1 := nt1+1..n1;   	# indices of the 1st coordinate of the validation set
set Tv_2 := nt1+1..n2;		# indices of the 2nd coordinates of the validation set

# variable values
param x1{i in T_1 union Tv_1};			# input values (first component)
param x2{j in T_2 union Tv_2};			# input values (second component)

# function values
param f{i in T_1 union Tv_1, j in T_2 union Tv_2};	# output values assuming f is scalar

# training set (input indices)
set T := {T_1,T_2};
set Tv := {Tv_1,Tv_2};

## neural network graph
param nI;
param nH;
param nO;

set I := 1 .. nI; 		# input nodes
set H := nI+1 .. nI+nH; # hidden nodes
set O := nI+nH+1 .. nI+nH+nO; # output nodes
set V := I union H union O;
set GIH := {I,H};
set GHO := {H,O};
set GIO := {I,O};
set G := GIH union GHO;
#set G; # adjacency matrix

# in-neighbourhoods
set N{i in V} := {j in V : (j,i) in G};

# big M
param M default 10e15;

#### decision variables ####
var v_plus{T union Tv,V} >= 0;					# positive post-activated values
var v_minus{T union Tv,V} >= 0;					# negative post-activated values
var z{T union Tv,V} >= 0, <= 1, default 1;		# relaxed binary variable to have v_plus <= 0 xor v_minus <= 0	
var A{G};								# weights NN
var b{V diff I};						# biases NN


#### objective function ####
minimize nnerr:
	sum{(i_x1,i_x2) in T union Tv, o in O} (v_plus[i_x1,i_x2,o] - v_minus[i_x1,i_x2,o] - f[i_x1,i_x2])^2;

#### constraints ####
# input
subject to input_x1{(i_x1,i_x2) in T}:
	v_plus[i_x1,i_x2,1] - v_minus[i_x1,i_x2,1] = x1[i_x1];
subject to input_x2{(i_x1,i_x2) in T}:
	v_plus[i_x1,i_x2,2] - v_minus[i_x1,i_x2,2] = x2[i_x2];

# hidden nodes with ReLU
subject to hidden_pre_upper{(i_x1,i_x2) in T, h in H}:
	(v_plus[i_x1,i_x2,h] - v_minus[i_x1,i_x2,h]) <= sum{i in N[h]} A[i,h]*v_plus[i_x1,i_x2,i] + b[h];

subject to hidden_pre_lower{(i_x1,i_x2) in T, h in H}:
	(v_plus[i_x1,i_x2,h] - v_minus[i_x1,i_x2,h]) >= (sum{i in N[h]} A[i,h]*v_plus[i_x1,i_x2,i] + b[h]);

subject to preactivation_pos{(i_x1,i_x2) in T, v in V}:
	v_plus[i_x1,i_x2,v] <= M*z[i_x1,i_x2,v];
	
subject to preactivation_neg{(i_x1,i_x2) in T, v in V}:
	v_minus[i_x1,i_x2,v] <= M*(1-z[i_x1,i_x2,v]);

# output node is linear
subject to output_upper{(i_x1,i_x2) in T, o in O}:
	(sum{i in N[o]} A[i,o]*(v_plus[i_x1,i_x2,i] - v_minus[i_x1,i_x2,i]) + b[o]) <= (v_plus[i_x1,i_x2,o] - v_minus[i_x1,i_x2,o]);
	
subject to output_lower{(i_x1,i_x2) in T, o in O}:
	(sum{i in N[o]} A[i,o]*(v_plus[i_x1,i_x2,i] - v_minus[i_x1,i_x2,i]) + b[o]) >= (v_plus[i_x1,i_x2,o] - v_minus[i_x1,i_x2,o]);