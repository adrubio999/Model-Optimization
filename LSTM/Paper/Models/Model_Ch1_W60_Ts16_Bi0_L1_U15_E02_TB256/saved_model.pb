??:
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
StatelessWhile

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.32v2.5.2-194-g959e9b2a0c08??8
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*&
shared_namelstm/lstm_cell/kernel

)lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/kernel*
_output_shapes

:<*
dtype0
?
lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*0
shared_name!lstm/lstm_cell/recurrent_kernel
?
3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/recurrent_kernel*
_output_shapes

:<*
dtype0
~
lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*$
shared_namelstm/lstm_cell/bias
w
'lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell/bias*
_output_shapes
:<*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/m
?
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:*
dtype0
?
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/m
?
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
?
Adam/lstm/lstm_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*-
shared_nameAdam/lstm/lstm_cell/kernel/m
?
0Adam/lstm/lstm_cell/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/kernel/m*
_output_shapes

:<*
dtype0
?
&Adam/lstm/lstm_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*7
shared_name(&Adam/lstm/lstm_cell/recurrent_kernel/m
?
:Adam/lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp&Adam/lstm/lstm_cell/recurrent_kernel/m*
_output_shapes

:<*
dtype0
?
Adam/lstm/lstm_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*+
shared_nameAdam/lstm/lstm_cell/bias/m
?
.Adam/lstm/lstm_cell/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/bias/m*
_output_shapes
:<*
dtype0
?
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/v
?
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:*
dtype0
?
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/v
?
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
?
Adam/lstm/lstm_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*-
shared_nameAdam/lstm/lstm_cell/kernel/v
?
0Adam/lstm/lstm_cell/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/kernel/v*
_output_shapes

:<*
dtype0
?
&Adam/lstm/lstm_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*7
shared_name(&Adam/lstm/lstm_cell/recurrent_kernel/v
?
:Adam/lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp&Adam/lstm/lstm_cell/recurrent_kernel/v*
_output_shapes

:<*
dtype0
?
Adam/lstm/lstm_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*+
shared_nameAdam/lstm/lstm_cell/bias/v
?
.Adam/lstm/lstm_cell/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/bias/v*
_output_shapes
:<*
dtype0

NoOpNoOp
?2
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?2
value?2B?2 B?2
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
	optimizer
	variables
	regularization_losses

trainable_variables
	keras_api

signatures
 
l
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
?
axis
	gamma
beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
R
 	variables
!regularization_losses
"trainable_variables
#	keras_api
h

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
?
*iter

+beta_1

,beta_2
	-decay
.learning_ratemfmg$mh%mi/mj0mk1mlvmvn$vo%vp/vq0vr1vs
?
/0
01
12
3
4
5
6
$7
%8
 
1
/0
01
12
3
4
$5
%6
?
2layer_regularization_losses
	variables
3non_trainable_variables
4layer_metrics
	regularization_losses
5metrics

trainable_variables

6layers
 
?
7
state_size

/kernel
0recurrent_kernel
1bias
8	variables
9regularization_losses
:trainable_variables
;	keras_api
 

/0
01
12
 

/0
01
12
?
<layer_regularization_losses
	variables
=non_trainable_variables

>states
?layer_metrics
regularization_losses
@metrics
trainable_variables

Alayers
 
 
 
?
Blayer_regularization_losses
Cnon_trainable_variables
	variables
Dlayer_metrics
Emetrics
regularization_losses

Flayers
trainable_variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
 

0
1
?
Glayer_regularization_losses
Hnon_trainable_variables
	variables
Ilayer_metrics
Jmetrics
regularization_losses

Klayers
trainable_variables
 
 
 
?
Llayer_regularization_losses
Mnon_trainable_variables
 	variables
Nlayer_metrics
Ometrics
!regularization_losses

Players
"trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
?
Qlayer_regularization_losses
Rnon_trainable_variables
&	variables
Slayer_metrics
Tmetrics
'regularization_losses

Ulayers
(trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUElstm/lstm_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUElstm/lstm_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUElstm/lstm_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 

V0
W1
*
0
1
2
3
4
5
 

/0
01
12
 

/0
01
12
?
Xlayer_regularization_losses
Ynon_trainable_variables
8	variables
Zlayer_metrics
[metrics
9regularization_losses

\layers
:trainable_variables
 
 
 
 
 

0
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	]total
	^count
_	variables
`	keras_api
D
	atotal
	bcount
c
_fn_kwargs
d	variables
e	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

]0
^1

_	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

a0
b1

d	variables
??
VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/lstm/lstm_cell/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/lstm/lstm_cell/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/lstm/lstm_cell/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/lstm/lstm_cell/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/lstm/lstm_cell/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/lstm/lstm_cell/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1lstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense/kernel
dense/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_695157
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp)lstm/lstm_cell/kernel/Read/ReadVariableOp3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp'lstm/lstm_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp0Adam/lstm/lstm_cell/kernel/m/Read/ReadVariableOp:Adam/lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOp.Adam/lstm/lstm_cell/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp0Adam/lstm/lstm_cell/kernel/v/Read/ReadVariableOp:Adam/lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOp.Adam/lstm/lstm_cell/bias/v/Read/ReadVariableOpConst*-
Tin&
$2"	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_698348
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/biastotalcounttotal_1count_1 Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/dense/kernel/mAdam/dense/bias/mAdam/lstm/lstm_cell/kernel/m&Adam/lstm/lstm_cell/recurrent_kernel/mAdam/lstm/lstm_cell/bias/m Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/dense/kernel/vAdam/dense/bias/vAdam/lstm/lstm_cell/kernel/v&Adam/lstm/lstm_cell/recurrent_kernel/vAdam/lstm/lstm_cell/bias/v*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_698454??7
?S
?
'__forward_gpu_lstm_with_fallback_696115

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:?????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:?????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_3dcebcae-8c80-4c1d-962a-8718205d4aa8*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_695940_696116*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?T
?
'__forward_gpu_lstm_with_fallback_697102

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:??????????????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:??????????????????:?????????:?????????:<:<:<*=
api_implements+)lstm_8d2ec9af-4917-4ad0-990a-8d3966677fc3*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_696927_697103*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?S
?
'__forward_gpu_lstm_with_fallback_691856

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:?????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:?????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_1109ec47-c15b-4e26-9c01-f618535fb6ce*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_691681_691857*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?	
?
$__inference_signature_wrapper_695157
input_1
unknown:<
	unknown_0:<
	unknown_1:<
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_6919042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?"
?
@__inference_lstm_layer_call_and_return_conditional_losses_697540

inputs.
read_readvariableop_resource:<0
read_1_readvariableop_resource:<,
read_2_readvariableop_resource:<

identity_3??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1?
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:<*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:<2

Identity?
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:<*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:<2

Identity_1?
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:<*
dtype02
Read_2/ReadVariableOph

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:<2

Identity_2?
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:?????????:?????????:?????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference_standard_lstm_6972672
PartitionedCall?

Identity_3IdentityPartitionedCall:output:1^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_3"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
@__inference_lstm_layer_call_and_return_conditional_losses_697975

inputs.
read_readvariableop_resource:<0
read_1_readvariableop_resource:<,
read_2_readvariableop_resource:<

identity_3??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1?
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:<*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:<2

Identity?
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:<*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:<2

Identity_1?
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:<*
dtype02
Read_2/ReadVariableOph

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:<2

Identity_2?
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:?????????:?????????:?????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference_standard_lstm_6977022
PartitionedCall?

Identity_3IdentityPartitionedCall:output:1^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_3"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?@
?
 __inference_standard_lstm_694694

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????<2
MatMulj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????<2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
add^
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:?????????<2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :?????????:?????????: : :<:<:<* 
_read_only_resource_inputs
 *
bodyR
while_body_694609*
condR
while_cond_694608*_
output_shapesN
L: : : : :?????????:?????????: : :<:<:<*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:?????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_21c69771-b438-47fb-9c6b-4c500860443c*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?
?
E__inference_RippleTry_layer_call_and_return_conditional_losses_695099
input_1
lstm_695075:<
lstm_695077:<
lstm_695079:<(
batch_normalization_695083:(
batch_normalization_695085:(
batch_normalization_695087:(
batch_normalization_695089:
dense_695093:
dense_695095:
identity??+batch_normalization/StatefulPartitionedCall?dense/StatefulPartitionedCall?lstm/StatefulPartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCallinput_1lstm_695075lstm_695077lstm_695079*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_6942932
lstm/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_6943062
dropout/PartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0batch_normalization_695083batch_normalization_695085batch_normalization_695087batch_normalization_695089*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_6943272-
+batch_normalization/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_6943422
dropout_1/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_695093dense_695095*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6943752
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
*__inference_RippleTry_layer_call_fn_695203

inputs
unknown:<
	unknown_0:<
	unknown_1:<
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*)
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_RippleTry_layer_call_and_return_conditional_losses_6950282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?,
?
while_body_695760
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul_1?
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
	while/add?
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:?????????<2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:?????????2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:?????????2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:?????????2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :?????????:?????????: : :<:<:<: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:<:$	 

_output_shapes

:<: 


_output_shapes
:<
?S
?
'__forward_gpu_lstm_with_fallback_694964

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:?????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:?????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_21c69771-b438-47fb-9c6b-4c500860443c*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_694789_694965*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?
?
%__inference_lstm_layer_call_fn_696213
inputs_0
unknown:<
	unknown_0:<
	unknown_1:<
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_6928002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
E__inference_RippleTry_layer_call_and_return_conditional_losses_694382

inputs
lstm_694294:<
lstm_694296:<
lstm_694298:<(
batch_normalization_694328:(
batch_normalization_694330:(
batch_normalization_694332:(
batch_normalization_694334:
dense_694376:
dense_694378:
identity??+batch_normalization/StatefulPartitionedCall?dense/StatefulPartitionedCall?lstm/StatefulPartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm_694294lstm_694296lstm_694298*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_6942932
lstm/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_6943062
dropout/PartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0batch_normalization_694328batch_normalization_694330batch_normalization_694332batch_normalization_694334*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_6943272-
+batch_normalization/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_6943422
dropout_1/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_694376dense_694378*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6943752
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?,
?
while_body_696747
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul_1?
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
	while/add?
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:?????????<2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:?????????2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:?????????2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:?????????2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :?????????:?????????: : :<:<:<: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:<:$	 

_output_shapes

:<: 


_output_shapes
:<
??
?
;__inference___backward_gpu_lstm_with_fallback_692165_692341
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0?
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :??????????????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????25
3gradients/strided_slice_grad/StridedSliceGrad/begin?
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end?
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :??????????????????*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutation?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :??????????????????2&
$gradients/transpose_9_grad/transpose?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape?
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2"
 gradients/Squeeze_1_grad/Reshape?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :??????????????????2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:??????????????????:?????????:?????????:?2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :??????????????????2$
"gradients/transpose_grad/transpose?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod?
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_1_grad/Shape?
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_1?
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_2?
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_3?
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_4?
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_5?
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_6?
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_7?
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_8?
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_9?
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_10?
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_11?
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_12?
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_13?
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_14?
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_15?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:2
gradients/concat_1_grad/Slice?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_1?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_2?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_3?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_4?
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_5?
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_6?
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_7?
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_8?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_9?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_10?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_11?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_12?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_13?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_14?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_15?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
gradients/Reshape_grad/Shape?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:2 
gradients/Reshape_grad/Reshape?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_1_grad/Reshape?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_2_grad/Reshape?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_3_grad/Reshape?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_4_grad/Reshape?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_5_grad/Reshape?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_6_grad/Reshape?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_7_grad/Shape?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_7_grad/Reshape?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_8_grad/Reshape?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_9_grad/Reshape?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_10_grad/Reshape?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_11_grad/Reshape?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_12_grad/Reshape?
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_13_grad/Shape?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_13_grad/Reshape?
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_14_grad/Shape?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_14_grad/Reshape?
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_15_grad/Shape?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_15_grad/Reshape?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_1_grad/transpose?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_2_grad/transpose?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_3_grad/transpose?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_4_grad/transpose?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_5_grad/transpose?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_6_grad/transpose?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_7_grad/transpose?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_8_grad/transpose?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes
:x2
gradients/split_2_grad/concat?
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_grad/concat?
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rank?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape_1?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffset?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice?
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice_1?
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_2s

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_3u

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_4q

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:<2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:??????????????????:?????????:?????????: :??????????????????::?????????:?????????::??????????????????:?????????:?????????:?::?????????:?????????: ::::::::: : : : *=
api_implements+)lstm_84ff0d2d-5e3a-4329-93ff-605791529804*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_692340*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????::6
4
_output_shapes"
 :??????????????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: ::6
4
_output_shapes"
 :??????????????????: 

_output_shapes
::1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:	

_output_shapes
:::
6
4
_output_shapes"
 :??????????????????:1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?H
?
)__inference_gpu_lstm_with_fallback_694114

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_1?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:?????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:?????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_6f89c239-b136-43c3-957e-6d1798a93e4e*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?
c
*__inference_dropout_1_layer_call_fn_698172

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_6944332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
while_cond_694608
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_694608___redundant_placeholder04
0while_while_cond_694608___redundant_placeholder14
0while_while_cond_694608___redundant_placeholder24
0while_while_cond_694608___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_694306

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_layer_call_fn_698054

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_6944852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
;__inference___backward_gpu_lstm_with_fallback_694789_694965
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:?????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????25
3gradients/strided_slice_grad/StridedSliceGrad/begin?
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end?
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:?????????*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutation?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:?????????2&
$gradients/transpose_9_grad/transpose?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape?
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2"
 gradients/Squeeze_1_grad/Reshape?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:?????????2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:?????????2$
"gradients/transpose_grad/transpose?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod?
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_1_grad/Shape?
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_1?
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_2?
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_3?
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_4?
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_5?
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_6?
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_7?
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_8?
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_9?
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_10?
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_11?
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_12?
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_13?
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_14?
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_15?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:2
gradients/concat_1_grad/Slice?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_1?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_2?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_3?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_4?
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_5?
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_6?
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_7?
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_8?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_9?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_10?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_11?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_12?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_13?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_14?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_15?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
gradients/Reshape_grad/Shape?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:2 
gradients/Reshape_grad/Reshape?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_1_grad/Reshape?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_2_grad/Reshape?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_3_grad/Reshape?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_4_grad/Reshape?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_5_grad/Reshape?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_6_grad/Reshape?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_7_grad/Shape?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_7_grad/Reshape?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_8_grad/Reshape?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_9_grad/Reshape?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_10_grad/Reshape?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_11_grad/Reshape?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_12_grad/Reshape?
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_13_grad/Shape?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_13_grad/Reshape?
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_14_grad/Shape?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_14_grad/Reshape?
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_15_grad/Shape?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_15_grad/Reshape?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_1_grad/transpose?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_2_grad/transpose?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_3_grad/transpose?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_4_grad/transpose?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_5_grad/transpose?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_6_grad/transpose?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_7_grad/transpose?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_8_grad/transpose?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes
:x2
gradients/split_2_grad/concat?
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_grad/concat?
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rank?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape_1?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffset?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice?
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice_1~
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_2s

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_3u

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_4q

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:<2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????: :?????????::?????????:?????????::?????????:?????????:?????????:?::?????????:?????????: ::::::::: : : : *=
api_implements+)lstm_21c69771-b438-47fb-9c6b-4c500860443c*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_694964*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :1-
+
_output_shapes
:?????????: 

_output_shapes
::1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:	

_output_shapes
::1
-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?S
?
'__forward_gpu_lstm_with_fallback_695635

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:?????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:?????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_333a86cd-1619-47ae-aeec-a18c90718136*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_695460_695636*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?	
?
while_cond_691984
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_691984___redundant_placeholder04
0while_while_cond_691984___redundant_placeholder14
0while_while_cond_691984___redundant_placeholder24
0while_while_cond_691984___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
?
&__inference_dense_layer_call_fn_698198

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6943752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?f
?
E__inference_RippleTry_layer_call_and_return_conditional_losses_695683

inputs3
!lstm_read_readvariableop_resource:<5
#lstm_read_1_readvariableop_resource:<1
#lstm_read_2_readvariableop_resource:<C
5batch_normalization_batchnorm_readvariableop_resource:G
9batch_normalization_batchnorm_mul_readvariableop_resource:E
7batch_normalization_batchnorm_readvariableop_1_resource:E
7batch_normalization_batchnorm_readvariableop_2_resource:9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity??,batch_normalization/batchnorm/ReadVariableOp?.batch_normalization/batchnorm/ReadVariableOp_1?.batch_normalization/batchnorm/ReadVariableOp_2?0batch_normalization/batchnorm/mul/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?lstm/Read/ReadVariableOp?lstm/Read_1/ReadVariableOp?lstm/Read_2/ReadVariableOpN

lstm/ShapeShapeinputs*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack?
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1?
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2?
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros/mul/y?
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros/packed/1?
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const?

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros_1/mul/y?
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm/zeros_1/Less/y?
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros_1/packed/1?
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const?
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm/zeros_1?
lstm/Read/ReadVariableOpReadVariableOp!lstm_read_readvariableop_resource*
_output_shapes

:<*
dtype02
lstm/Read/ReadVariableOpu
lstm/IdentityIdentity lstm/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:<2
lstm/Identity?
lstm/Read_1/ReadVariableOpReadVariableOp#lstm_read_1_readvariableop_resource*
_output_shapes

:<*
dtype02
lstm/Read_1/ReadVariableOp{
lstm/Identity_1Identity"lstm/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

:<2
lstm/Identity_1?
lstm/Read_2/ReadVariableOpReadVariableOp#lstm_read_2_readvariableop_resource*
_output_shapes
:<*
dtype02
lstm/Read_2/ReadVariableOpw
lstm/Identity_2Identity"lstm/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes
:<2
lstm/Identity_2?
lstm/PartitionedCallPartitionedCallinputslstm/zeros:output:0lstm/zeros_1:output:0lstm/Identity:output:0lstm/Identity_1:output:0lstm/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:?????????:?????????:?????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference_standard_lstm_6953652
lstm/PartitionedCall?
dropout/IdentityIdentitylstm/PartitionedCall:output:1*
T0*+
_output_shapes
:?????????2
dropout/Identity?
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization/batchnorm/ReadVariableOp?
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2%
#batch_normalization/batchnorm/add/y?
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/add?
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/Rsqrt?
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/mul?
#batch_normalization/batchnorm/mul_1Muldropout/Identity:output:0%batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2%
#batch_normalization/batchnorm/mul_1?
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1?
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/mul_2?
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2?
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/sub?
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2%
#batch_normalization/batchnorm/add_1?
dropout_1/IdentityIdentity'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
dropout_1/Identity?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freey
dense/Tensordot/ShapeShapedropout_1/Identity:output:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transposedropout_1/Identity:output:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense/BiasAddw
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
dense/Sigmoid?
IdentityIdentitydense/Sigmoid:y:0-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^lstm/Read/ReadVariableOp^lstm/Read_1/ReadVariableOp^lstm/Read_2/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp24
lstm/Read/ReadVariableOplstm/Read/ReadVariableOp28
lstm/Read_1/ReadVariableOplstm/Read_1/ReadVariableOp28
lstm/Read_2/ReadVariableOplstm/Read_2/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?T
?
'__forward_gpu_lstm_with_fallback_692797

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:??????????????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:??????????????????:?????????:?????????:<:<:<*=
api_implements+)lstm_55cb023b-f9e0-4eff-9ca0-fbc632457c5d*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_692622_692798*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
??
?
;__inference___backward_gpu_lstm_with_fallback_695460_695636
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:?????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????25
3gradients/strided_slice_grad/StridedSliceGrad/begin?
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end?
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:?????????*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutation?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:?????????2&
$gradients/transpose_9_grad/transpose?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape?
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2"
 gradients/Squeeze_1_grad/Reshape?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:?????????2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:?????????2$
"gradients/transpose_grad/transpose?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod?
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_1_grad/Shape?
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_1?
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_2?
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_3?
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_4?
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_5?
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_6?
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_7?
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_8?
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_9?
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_10?
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_11?
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_12?
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_13?
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_14?
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_15?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:2
gradients/concat_1_grad/Slice?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_1?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_2?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_3?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_4?
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_5?
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_6?
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_7?
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_8?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_9?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_10?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_11?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_12?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_13?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_14?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_15?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
gradients/Reshape_grad/Shape?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:2 
gradients/Reshape_grad/Reshape?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_1_grad/Reshape?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_2_grad/Reshape?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_3_grad/Reshape?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_4_grad/Reshape?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_5_grad/Reshape?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_6_grad/Reshape?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_7_grad/Shape?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_7_grad/Reshape?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_8_grad/Reshape?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_9_grad/Reshape?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_10_grad/Reshape?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_11_grad/Reshape?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_12_grad/Reshape?
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_13_grad/Shape?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_13_grad/Reshape?
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_14_grad/Shape?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_14_grad/Reshape?
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_15_grad/Shape?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_15_grad/Reshape?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_1_grad/transpose?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_2_grad/transpose?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_3_grad/transpose?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_4_grad/transpose?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_5_grad/transpose?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_6_grad/transpose?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_7_grad/transpose?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_8_grad/transpose?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes
:x2
gradients/split_2_grad/concat?
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_grad/concat?
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rank?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape_1?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffset?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice?
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice_1~
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_2s

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_3u

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_4q

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:<2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????: :?????????::?????????:?????????::?????????:?????????:?????????:?::?????????:?????????: ::::::::: : : : *=
api_implements+)lstm_333a86cd-1619-47ae-aeec-a18c90718136*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_695635*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :1-
+
_output_shapes
:?????????: 

_output_shapes
::1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:	

_output_shapes
::1
-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
while_cond_693934
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_693934___redundant_placeholder04
0while_while_cond_693934___redundant_placeholder14
0while_while_cond_693934___redundant_placeholder24
0while_while_cond_693934___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_694327

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_698128

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?A
?
 __inference_standard_lstm_696397

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????<2
MatMulj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????<2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
add^
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:?????????<2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :?????????:?????????: : :<:<:<* 
_read_only_resource_inputs
 *
bodyR
while_body_696312*
condR
while_cond_696311*_
output_shapesN
L: : : : :?????????:?????????: : :<:<:<*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:??????????????????:?????????:?????????:<:<:<*=
api_implements+)lstm_10775f66-0d42-44f9-b303-bd1235de0720*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_698074

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
??
?
;__inference___backward_gpu_lstm_with_fallback_696927_697103
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0?
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :??????????????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????25
3gradients/strided_slice_grad/StridedSliceGrad/begin?
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end?
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :??????????????????*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutation?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :??????????????????2&
$gradients/transpose_9_grad/transpose?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape?
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2"
 gradients/Squeeze_1_grad/Reshape?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :??????????????????2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:??????????????????:?????????:?????????:?2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :??????????????????2$
"gradients/transpose_grad/transpose?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod?
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_1_grad/Shape?
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_1?
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_2?
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_3?
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_4?
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_5?
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_6?
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_7?
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_8?
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_9?
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_10?
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_11?
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_12?
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_13?
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_14?
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_15?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:2
gradients/concat_1_grad/Slice?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_1?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_2?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_3?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_4?
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_5?
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_6?
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_7?
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_8?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_9?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_10?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_11?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_12?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_13?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_14?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_15?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
gradients/Reshape_grad/Shape?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:2 
gradients/Reshape_grad/Reshape?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_1_grad/Reshape?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_2_grad/Reshape?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_3_grad/Reshape?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_4_grad/Reshape?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_5_grad/Reshape?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_6_grad/Reshape?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_7_grad/Shape?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_7_grad/Reshape?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_8_grad/Reshape?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_9_grad/Reshape?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_10_grad/Reshape?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_11_grad/Reshape?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_12_grad/Reshape?
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_13_grad/Shape?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_13_grad/Reshape?
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_14_grad/Shape?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_14_grad/Reshape?
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_15_grad/Shape?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_15_grad/Reshape?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_1_grad/transpose?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_2_grad/transpose?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_3_grad/transpose?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_4_grad/transpose?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_5_grad/transpose?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_6_grad/transpose?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_7_grad/transpose?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_8_grad/transpose?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes
:x2
gradients/split_2_grad/concat?
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_grad/concat?
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rank?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape_1?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffset?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice?
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice_1?
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_2s

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_3u

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_4q

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:<2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:??????????????????:?????????:?????????: :??????????????????::?????????:?????????::??????????????????:?????????:?????????:?::?????????:?????????: ::::::::: : : : *=
api_implements+)lstm_8d2ec9af-4917-4ad0-990a-8d3966677fc3*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_697102*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????::6
4
_output_shapes"
 :??????????????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: ::6
4
_output_shapes"
 :??????????????????: 

_output_shapes
::1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:	

_output_shapes
:::
6
4
_output_shapes"
 :??????????????????:1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?@
?
 __inference_standard_lstm_691586

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????<2
MatMulj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????<2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
add^
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:?????????<2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :?????????:?????????: : :<:<:<* 
_read_only_resource_inputs
 *
bodyR
while_body_691501*
condR
while_cond_691500*_
output_shapesN
L: : : : :?????????:?????????: : :<:<:<*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:?????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_1109ec47-c15b-4e26-9c01-f618535fb6ce*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?	
?
*__inference_RippleTry_layer_call_fn_695180

inputs
unknown:<
	unknown_0:<
	unknown_1:<
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_RippleTry_layer_call_and_return_conditional_losses_6943822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_694433

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0*
seed???)*
seed22&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
;__inference___backward_gpu_lstm_with_fallback_697362_697538
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:?????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????25
3gradients/strided_slice_grad/StridedSliceGrad/begin?
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end?
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:?????????*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutation?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:?????????2&
$gradients/transpose_9_grad/transpose?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape?
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2"
 gradients/Squeeze_1_grad/Reshape?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:?????????2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:?????????2$
"gradients/transpose_grad/transpose?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod?
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_1_grad/Shape?
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_1?
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_2?
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_3?
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_4?
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_5?
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_6?
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_7?
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_8?
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_9?
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_10?
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_11?
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_12?
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_13?
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_14?
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_15?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:2
gradients/concat_1_grad/Slice?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_1?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_2?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_3?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_4?
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_5?
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_6?
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_7?
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_8?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_9?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_10?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_11?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_12?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_13?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_14?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_15?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
gradients/Reshape_grad/Shape?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:2 
gradients/Reshape_grad/Reshape?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_1_grad/Reshape?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_2_grad/Reshape?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_3_grad/Reshape?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_4_grad/Reshape?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_5_grad/Reshape?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_6_grad/Reshape?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_7_grad/Shape?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_7_grad/Reshape?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_8_grad/Reshape?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_9_grad/Reshape?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_10_grad/Reshape?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_11_grad/Reshape?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_12_grad/Reshape?
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_13_grad/Shape?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_13_grad/Reshape?
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_14_grad/Shape?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_14_grad/Reshape?
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_15_grad/Shape?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_15_grad/Reshape?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_1_grad/transpose?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_2_grad/transpose?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_3_grad/transpose?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_4_grad/transpose?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_5_grad/transpose?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_6_grad/transpose?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_7_grad/transpose?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_8_grad/transpose?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes
:x2
gradients/split_2_grad/concat?
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_grad/concat?
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rank?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape_1?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffset?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice?
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice_1~
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_2s

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_3u

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_4q

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:<2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????: :?????????::?????????:?????????::?????????:?????????:?????????:?::?????????:?????????: ::::::::: : : : *=
api_implements+)lstm_5796cf19-f097-407f-ba62-351741090953*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_697537*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :1-
+
_output_shapes
:?????????: 

_output_shapes
::1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:	

_output_shapes
::1
-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?@
?
 __inference_standard_lstm_697702

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????<2
MatMulj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????<2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
add^
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:?????????<2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :?????????:?????????: : :<:<:<* 
_read_only_resource_inputs
 *
bodyR
while_body_697617*
condR
while_cond_697616*_
output_shapesN
L: : : : :?????????:?????????: : :<:<:<*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:?????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_d31ccf94-a07e-4964-8059-939fe35bedbb*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?
?
E__inference_RippleTry_layer_call_and_return_conditional_losses_695126
input_1
lstm_695102:<
lstm_695104:<
lstm_695106:<(
batch_normalization_695110:(
batch_normalization_695112:(
batch_normalization_695114:(
batch_normalization_695116:
dense_695120:
dense_695122:
identity??+batch_normalization/StatefulPartitionedCall?dense/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?lstm/StatefulPartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCallinput_1lstm_695102lstm_695104lstm_695106*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_6949672
lstm/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_6945162!
dropout/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0batch_normalization_695110batch_normalization_695112batch_normalization_695114batch_normalization_695116*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_6944852-
+batch_normalization/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_6944332#
!dropout_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_695120dense_695122*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6943752
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?"
?
@__inference_lstm_layer_call_and_return_conditional_losses_694293

inputs.
read_readvariableop_resource:<0
read_1_readvariableop_resource:<,
read_2_readvariableop_resource:<

identity_3??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1?
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:<*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:<2

Identity?
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:<*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:<2

Identity_1?
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:<*
dtype02
Read_2/ReadVariableOph

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:<2

Identity_2?
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:?????????:?????????:?????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference_standard_lstm_6940202
PartitionedCall?

Identity_3IdentityPartitionedCall:output:1^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_3"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
E__inference_RippleTry_layer_call_and_return_conditional_losses_696191

inputs3
!lstm_read_readvariableop_resource:<5
#lstm_read_1_readvariableop_resource:<1
#lstm_read_2_readvariableop_resource:<I
;batch_normalization_assignmovingavg_readvariableop_resource:K
=batch_normalization_assignmovingavg_1_readvariableop_resource:G
9batch_normalization_batchnorm_mul_readvariableop_resource:C
5batch_normalization_batchnorm_readvariableop_resource:9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity??#batch_normalization/AssignMovingAvg?2batch_normalization/AssignMovingAvg/ReadVariableOp?%batch_normalization/AssignMovingAvg_1?4batch_normalization/AssignMovingAvg_1/ReadVariableOp?,batch_normalization/batchnorm/ReadVariableOp?0batch_normalization/batchnorm/mul/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?lstm/Read/ReadVariableOp?lstm/Read_1/ReadVariableOp?lstm/Read_2/ReadVariableOpN

lstm/ShapeShapeinputs*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack?
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1?
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2?
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros/mul/y?
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros/packed/1?
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const?

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros_1/mul/y?
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm/zeros_1/Less/y?
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros_1/packed/1?
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const?
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm/zeros_1?
lstm/Read/ReadVariableOpReadVariableOp!lstm_read_readvariableop_resource*
_output_shapes

:<*
dtype02
lstm/Read/ReadVariableOpu
lstm/IdentityIdentity lstm/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:<2
lstm/Identity?
lstm/Read_1/ReadVariableOpReadVariableOp#lstm_read_1_readvariableop_resource*
_output_shapes

:<*
dtype02
lstm/Read_1/ReadVariableOp{
lstm/Identity_1Identity"lstm/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

:<2
lstm/Identity_1?
lstm/Read_2/ReadVariableOpReadVariableOp#lstm_read_2_readvariableop_resource*
_output_shapes
:<*
dtype02
lstm/Read_2/ReadVariableOpw
lstm/Identity_2Identity"lstm/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes
:<2
lstm/Identity_2?
lstm/PartitionedCallPartitionedCallinputslstm/zeros:output:0lstm/zeros_1:output:0lstm/Identity:output:0lstm/Identity_1:output:0lstm/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:?????????:?????????:?????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference_standard_lstm_6958452
lstm/PartitionedCalls
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Const?
dropout/dropout/MulMullstm/PartitionedCall:output:1dropout/dropout/Const:output:0*
T0*+
_output_shapes
:?????????2
dropout/dropout/Mul{
dropout/dropout/ShapeShapelstm/PartitionedCall:output:1*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0*
seed???)*
seed22.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????2
dropout/dropout/Mul_1?
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       24
2batch_normalization/moments/mean/reduction_indices?
 batch_normalization/moments/meanMeandropout/dropout/Mul_1:z:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2"
 batch_normalization/moments/mean?
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:2*
(batch_normalization/moments/StopGradient?
-batch_normalization/moments/SquaredDifferenceSquaredDifferencedropout/dropout/Mul_1:z:01batch_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????2/
-batch_normalization/moments/SquaredDifference?
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       28
6batch_normalization/moments/variance/reduction_indices?
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2&
$batch_normalization/moments/variance?
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze?
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1?
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)batch_normalization/AssignMovingAvg/decay?
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp?
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:2)
'batch_normalization/AssignMovingAvg/sub?
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2)
'batch_normalization/AssignMovingAvg/mul?
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02%
#batch_normalization/AssignMovingAvg?
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization/AssignMovingAvg_1/decay?
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp?
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2+
)batch_normalization/AssignMovingAvg_1/sub?
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2+
)batch_normalization/AssignMovingAvg_1/mul?
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization/AssignMovingAvg_1?
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2%
#batch_normalization/batchnorm/add/y?
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/add?
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/Rsqrt?
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/mul?
#batch_normalization/batchnorm/mul_1Muldropout/dropout/Mul_1:z:0%batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2%
#batch_normalization/batchnorm/mul_1?
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/mul_2?
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization/batchnorm/ReadVariableOp?
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/sub?
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2%
#batch_normalization/batchnorm/add_1w
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_1/dropout/Const?
dropout_1/dropout/MulMul'batch_normalization/batchnorm/add_1:z:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeShape'batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0*
seed???)*
seed220
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????2
dropout_1/dropout/Mul_1?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freey
dense/Tensordot/ShapeShapedropout_1/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transposedropout_1/dropout/Mul_1:z:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense/BiasAddw
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
dense/Sigmoid?
IdentityIdentitydense/Sigmoid:y:0$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^lstm/Read/ReadVariableOp^lstm/Read_1/ReadVariableOp^lstm/Read_2/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp24
lstm/Read/ReadVariableOplstm/Read/ReadVariableOp28
lstm/Read_1/ReadVariableOplstm/Read_1/ReadVariableOp28
lstm/Read_2/ReadVariableOplstm/Read_2/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
*__inference_RippleTry_layer_call_fn_695072
input_1
unknown:<
	unknown_0:<
	unknown_1:<
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*)
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_RippleTry_layer_call_and_return_conditional_losses_6950282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?H
?
)__inference_gpu_lstm_with_fallback_697361

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_1?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:?????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:?????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_5796cf19-f097-407f-ba62-351741090953*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?	
?
while_cond_696746
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_696746___redundant_placeholder04
0while_while_cond_696746___redundant_placeholder14
0while_while_cond_696746___redundant_placeholder24
0while_while_cond_696746___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?	
?
while_cond_692441
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_692441___redundant_placeholder04
0while_while_cond_692441___redundant_placeholder14
0while_while_cond_692441___redundant_placeholder24
0while_while_cond_692441___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?	
?
while_cond_696311
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_696311___redundant_placeholder04
0while_while_cond_696311___redundant_placeholder14
0while_while_cond_696311___redundant_placeholder24
0while_while_cond_696311___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?@
?
 __inference_standard_lstm_694020

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????<2
MatMulj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????<2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
add^
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:?????????<2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :?????????:?????????: : :<:<:<* 
_read_only_resource_inputs
 *
bodyR
while_body_693935*
condR
while_cond_693934*_
output_shapesN
L: : : : :?????????:?????????: : :<:<:<*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:?????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_6f89c239-b136-43c3-957e-6d1798a93e4e*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?@
?
 __inference_standard_lstm_695845

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????<2
MatMulj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????<2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
add^
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:?????????<2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :?????????:?????????: : :<:<:<* 
_read_only_resource_inputs
 *
bodyR
while_body_695760*
condR
while_cond_695759*_
output_shapesN
L: : : : :?????????:?????????: : :<:<:<*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:?????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_3dcebcae-8c80-4c1d-962a-8718205d4aa8*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?	
?
while_cond_695759
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_695759___redundant_placeholder04
0while_while_cond_695759___redundant_placeholder14
0while_while_cond_695759___redundant_placeholder24
0while_while_cond_695759___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?*
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_698162

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_698177

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
;__inference___backward_gpu_lstm_with_fallback_695940_696116
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:?????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????25
3gradients/strided_slice_grad/StridedSliceGrad/begin?
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end?
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:?????????*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutation?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:?????????2&
$gradients/transpose_9_grad/transpose?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape?
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2"
 gradients/Squeeze_1_grad/Reshape?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:?????????2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:?????????2$
"gradients/transpose_grad/transpose?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod?
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_1_grad/Shape?
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_1?
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_2?
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_3?
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_4?
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_5?
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_6?
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_7?
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_8?
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_9?
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_10?
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_11?
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_12?
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_13?
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_14?
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_15?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:2
gradients/concat_1_grad/Slice?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_1?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_2?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_3?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_4?
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_5?
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_6?
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_7?
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_8?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_9?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_10?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_11?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_12?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_13?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_14?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_15?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
gradients/Reshape_grad/Shape?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:2 
gradients/Reshape_grad/Reshape?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_1_grad/Reshape?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_2_grad/Reshape?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_3_grad/Reshape?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_4_grad/Reshape?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_5_grad/Reshape?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_6_grad/Reshape?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_7_grad/Shape?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_7_grad/Reshape?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_8_grad/Reshape?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_9_grad/Reshape?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_10_grad/Reshape?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_11_grad/Reshape?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_12_grad/Reshape?
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_13_grad/Shape?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_13_grad/Reshape?
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_14_grad/Shape?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_14_grad/Reshape?
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_15_grad/Shape?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_15_grad/Reshape?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_1_grad/transpose?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_2_grad/transpose?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_3_grad/transpose?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_4_grad/transpose?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_5_grad/transpose?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_6_grad/transpose?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_7_grad/transpose?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_8_grad/transpose?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes
:x2
gradients/split_2_grad/concat?
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_grad/concat?
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rank?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape_1?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffset?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice?
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice_1~
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_2s

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_3u

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_4q

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:<2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????: :?????????::?????????:?????????::?????????:?????????:?????????:?::?????????:?????????: ::::::::: : : : *=
api_implements+)lstm_3dcebcae-8c80-4c1d-962a-8718205d4aa8*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_696115*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :1-
+
_output_shapes
:?????????: 

_output_shapes
::1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:	

_output_shapes
::1
-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_694342

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_lstm_layer_call_fn_696224

inputs
unknown:<
	unknown_0:<
	unknown_1:<
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_6942932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
while_cond_695279
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_695279___redundant_placeholder04
0while_while_cond_695279___redundant_placeholder14
0while_while_cond_695279___redundant_placeholder24
0while_while_cond_695279___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?*
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_698108

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
E__inference_RippleTry_layer_call_and_return_conditional_losses_695028

inputs
lstm_695004:<
lstm_695006:<
lstm_695008:<(
batch_normalization_695012:(
batch_normalization_695014:(
batch_normalization_695016:(
batch_normalization_695018:
dense_695022:
dense_695024:
identity??+batch_normalization/StatefulPartitionedCall?dense/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?lstm/StatefulPartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm_695004lstm_695006lstm_695008*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_6949672
lstm/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_6945162!
dropout/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0batch_normalization_695012batch_normalization_695014batch_normalization_695016batch_normalization_695018*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_6944852-
+batch_normalization/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_6944332#
!dropout_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_695022dense_695024*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6943752
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?H
?
)__inference_gpu_lstm_with_fallback_692621

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_1?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:??????????????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:??????????????????:?????????:?????????:<:<:<*=
api_implements+)lstm_55cb023b-f9e0-4eff-9ca0-fbc632457c5d*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?,
?
while_body_697617
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul_1?
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
	while/add?
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:?????????<2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:?????????2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:?????????2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:?????????2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :?????????:?????????: : :<:<:<: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:<:$	 

_output_shapes

:<: 


_output_shapes
:<
?H
?
)__inference_gpu_lstm_with_fallback_697796

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_1?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:?????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:?????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_d31ccf94-a07e-4964-8059-939fe35bedbb*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?@
?
 __inference_standard_lstm_697267

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????<2
MatMulj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????<2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
add^
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:?????????<2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :?????????:?????????: : :<:<:<* 
_read_only_resource_inputs
 *
bodyR
while_body_697182*
condR
while_cond_697181*_
output_shapesN
L: : : : :?????????:?????????: : :<:<:<*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:?????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_5796cf19-f097-407f-ba62-351741090953*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?	
?
while_cond_691500
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_691500___redundant_placeholder04
0while_while_cond_691500___redundant_placeholder14
0while_while_cond_691500___redundant_placeholder24
0while_while_cond_691500___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?,
?
while_body_696312
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul_1?
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
	while/add?
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:?????????<2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:?????????2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:?????????2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:?????????2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :?????????:?????????: : :<:<:<: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:<:$	 

_output_shapes

:<: 


_output_shapes
:<
?,
?
while_body_692442
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul_1?
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
	while/add?
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:?????????<2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:?????????2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:?????????2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:?????????2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :?????????:?????????: : :<:<:<: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:<:$	 

_output_shapes

:<: 


_output_shapes
:<
?{
?
!__inference__wrapped_model_691904
input_1=
+rippletry_lstm_read_readvariableop_resource:<?
-rippletry_lstm_read_1_readvariableop_resource:<;
-rippletry_lstm_read_2_readvariableop_resource:<M
?rippletry_batch_normalization_batchnorm_readvariableop_resource:Q
Crippletry_batch_normalization_batchnorm_mul_readvariableop_resource:O
Arippletry_batch_normalization_batchnorm_readvariableop_1_resource:O
Arippletry_batch_normalization_batchnorm_readvariableop_2_resource:C
1rippletry_dense_tensordot_readvariableop_resource:=
/rippletry_dense_biasadd_readvariableop_resource:
identity??6RippleTry/batch_normalization/batchnorm/ReadVariableOp?8RippleTry/batch_normalization/batchnorm/ReadVariableOp_1?8RippleTry/batch_normalization/batchnorm/ReadVariableOp_2?:RippleTry/batch_normalization/batchnorm/mul/ReadVariableOp?&RippleTry/dense/BiasAdd/ReadVariableOp?(RippleTry/dense/Tensordot/ReadVariableOp?"RippleTry/lstm/Read/ReadVariableOp?$RippleTry/lstm/Read_1/ReadVariableOp?$RippleTry/lstm/Read_2/ReadVariableOpc
RippleTry/lstm/ShapeShapeinput_1*
T0*
_output_shapes
:2
RippleTry/lstm/Shape?
"RippleTry/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"RippleTry/lstm/strided_slice/stack?
$RippleTry/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$RippleTry/lstm/strided_slice/stack_1?
$RippleTry/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$RippleTry/lstm/strided_slice/stack_2?
RippleTry/lstm/strided_sliceStridedSliceRippleTry/lstm/Shape:output:0+RippleTry/lstm/strided_slice/stack:output:0-RippleTry/lstm/strided_slice/stack_1:output:0-RippleTry/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
RippleTry/lstm/strided_slicez
RippleTry/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
RippleTry/lstm/zeros/mul/y?
RippleTry/lstm/zeros/mulMul%RippleTry/lstm/strided_slice:output:0#RippleTry/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
RippleTry/lstm/zeros/mul}
RippleTry/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
RippleTry/lstm/zeros/Less/y?
RippleTry/lstm/zeros/LessLessRippleTry/lstm/zeros/mul:z:0$RippleTry/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
RippleTry/lstm/zeros/Less?
RippleTry/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
RippleTry/lstm/zeros/packed/1?
RippleTry/lstm/zeros/packedPack%RippleTry/lstm/strided_slice:output:0&RippleTry/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
RippleTry/lstm/zeros/packed}
RippleTry/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
RippleTry/lstm/zeros/Const?
RippleTry/lstm/zerosFill$RippleTry/lstm/zeros/packed:output:0#RippleTry/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
RippleTry/lstm/zeros~
RippleTry/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
RippleTry/lstm/zeros_1/mul/y?
RippleTry/lstm/zeros_1/mulMul%RippleTry/lstm/strided_slice:output:0%RippleTry/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
RippleTry/lstm/zeros_1/mul?
RippleTry/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
RippleTry/lstm/zeros_1/Less/y?
RippleTry/lstm/zeros_1/LessLessRippleTry/lstm/zeros_1/mul:z:0&RippleTry/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
RippleTry/lstm/zeros_1/Less?
RippleTry/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2!
RippleTry/lstm/zeros_1/packed/1?
RippleTry/lstm/zeros_1/packedPack%RippleTry/lstm/strided_slice:output:0(RippleTry/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
RippleTry/lstm/zeros_1/packed?
RippleTry/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
RippleTry/lstm/zeros_1/Const?
RippleTry/lstm/zeros_1Fill&RippleTry/lstm/zeros_1/packed:output:0%RippleTry/lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
RippleTry/lstm/zeros_1?
"RippleTry/lstm/Read/ReadVariableOpReadVariableOp+rippletry_lstm_read_readvariableop_resource*
_output_shapes

:<*
dtype02$
"RippleTry/lstm/Read/ReadVariableOp?
RippleTry/lstm/IdentityIdentity*RippleTry/lstm/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:<2
RippleTry/lstm/Identity?
$RippleTry/lstm/Read_1/ReadVariableOpReadVariableOp-rippletry_lstm_read_1_readvariableop_resource*
_output_shapes

:<*
dtype02&
$RippleTry/lstm/Read_1/ReadVariableOp?
RippleTry/lstm/Identity_1Identity,RippleTry/lstm/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

:<2
RippleTry/lstm/Identity_1?
$RippleTry/lstm/Read_2/ReadVariableOpReadVariableOp-rippletry_lstm_read_2_readvariableop_resource*
_output_shapes
:<*
dtype02&
$RippleTry/lstm/Read_2/ReadVariableOp?
RippleTry/lstm/Identity_2Identity,RippleTry/lstm/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes
:<2
RippleTry/lstm/Identity_2?
RippleTry/lstm/PartitionedCallPartitionedCallinput_1RippleTry/lstm/zeros:output:0RippleTry/lstm/zeros_1:output:0 RippleTry/lstm/Identity:output:0"RippleTry/lstm/Identity_1:output:0"RippleTry/lstm/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:?????????:?????????:?????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference_standard_lstm_6915862 
RippleTry/lstm/PartitionedCall?
RippleTry/dropout/IdentityIdentity'RippleTry/lstm/PartitionedCall:output:1*
T0*+
_output_shapes
:?????????2
RippleTry/dropout/Identity?
6RippleTry/batch_normalization/batchnorm/ReadVariableOpReadVariableOp?rippletry_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype028
6RippleTry/batch_normalization/batchnorm/ReadVariableOp?
-RippleTry/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2/
-RippleTry/batch_normalization/batchnorm/add/y?
+RippleTry/batch_normalization/batchnorm/addAddV2>RippleTry/batch_normalization/batchnorm/ReadVariableOp:value:06RippleTry/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2-
+RippleTry/batch_normalization/batchnorm/add?
-RippleTry/batch_normalization/batchnorm/RsqrtRsqrt/RippleTry/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2/
-RippleTry/batch_normalization/batchnorm/Rsqrt?
:RippleTry/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpCrippletry_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02<
:RippleTry/batch_normalization/batchnorm/mul/ReadVariableOp?
+RippleTry/batch_normalization/batchnorm/mulMul1RippleTry/batch_normalization/batchnorm/Rsqrt:y:0BRippleTry/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+RippleTry/batch_normalization/batchnorm/mul?
-RippleTry/batch_normalization/batchnorm/mul_1Mul#RippleTry/dropout/Identity:output:0/RippleTry/batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2/
-RippleTry/batch_normalization/batchnorm/mul_1?
8RippleTry/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpArippletry_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8RippleTry/batch_normalization/batchnorm/ReadVariableOp_1?
-RippleTry/batch_normalization/batchnorm/mul_2Mul@RippleTry/batch_normalization/batchnorm/ReadVariableOp_1:value:0/RippleTry/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2/
-RippleTry/batch_normalization/batchnorm/mul_2?
8RippleTry/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpArippletry_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02:
8RippleTry/batch_normalization/batchnorm/ReadVariableOp_2?
+RippleTry/batch_normalization/batchnorm/subSub@RippleTry/batch_normalization/batchnorm/ReadVariableOp_2:value:01RippleTry/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2-
+RippleTry/batch_normalization/batchnorm/sub?
-RippleTry/batch_normalization/batchnorm/add_1AddV21RippleTry/batch_normalization/batchnorm/mul_1:z:0/RippleTry/batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2/
-RippleTry/batch_normalization/batchnorm/add_1?
RippleTry/dropout_1/IdentityIdentity1RippleTry/batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
RippleTry/dropout_1/Identity?
(RippleTry/dense/Tensordot/ReadVariableOpReadVariableOp1rippletry_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02*
(RippleTry/dense/Tensordot/ReadVariableOp?
RippleTry/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2 
RippleTry/dense/Tensordot/axes?
RippleTry/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2 
RippleTry/dense/Tensordot/free?
RippleTry/dense/Tensordot/ShapeShape%RippleTry/dropout_1/Identity:output:0*
T0*
_output_shapes
:2!
RippleTry/dense/Tensordot/Shape?
'RippleTry/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'RippleTry/dense/Tensordot/GatherV2/axis?
"RippleTry/dense/Tensordot/GatherV2GatherV2(RippleTry/dense/Tensordot/Shape:output:0'RippleTry/dense/Tensordot/free:output:00RippleTry/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"RippleTry/dense/Tensordot/GatherV2?
)RippleTry/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)RippleTry/dense/Tensordot/GatherV2_1/axis?
$RippleTry/dense/Tensordot/GatherV2_1GatherV2(RippleTry/dense/Tensordot/Shape:output:0'RippleTry/dense/Tensordot/axes:output:02RippleTry/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$RippleTry/dense/Tensordot/GatherV2_1?
RippleTry/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
RippleTry/dense/Tensordot/Const?
RippleTry/dense/Tensordot/ProdProd+RippleTry/dense/Tensordot/GatherV2:output:0(RippleTry/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2 
RippleTry/dense/Tensordot/Prod?
!RippleTry/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!RippleTry/dense/Tensordot/Const_1?
 RippleTry/dense/Tensordot/Prod_1Prod-RippleTry/dense/Tensordot/GatherV2_1:output:0*RippleTry/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2"
 RippleTry/dense/Tensordot/Prod_1?
%RippleTry/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%RippleTry/dense/Tensordot/concat/axis?
 RippleTry/dense/Tensordot/concatConcatV2'RippleTry/dense/Tensordot/free:output:0'RippleTry/dense/Tensordot/axes:output:0.RippleTry/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 RippleTry/dense/Tensordot/concat?
RippleTry/dense/Tensordot/stackPack'RippleTry/dense/Tensordot/Prod:output:0)RippleTry/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2!
RippleTry/dense/Tensordot/stack?
#RippleTry/dense/Tensordot/transpose	Transpose%RippleTry/dropout_1/Identity:output:0)RippleTry/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2%
#RippleTry/dense/Tensordot/transpose?
!RippleTry/dense/Tensordot/ReshapeReshape'RippleTry/dense/Tensordot/transpose:y:0(RippleTry/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2#
!RippleTry/dense/Tensordot/Reshape?
 RippleTry/dense/Tensordot/MatMulMatMul*RippleTry/dense/Tensordot/Reshape:output:00RippleTry/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 RippleTry/dense/Tensordot/MatMul?
!RippleTry/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!RippleTry/dense/Tensordot/Const_2?
'RippleTry/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'RippleTry/dense/Tensordot/concat_1/axis?
"RippleTry/dense/Tensordot/concat_1ConcatV2+RippleTry/dense/Tensordot/GatherV2:output:0*RippleTry/dense/Tensordot/Const_2:output:00RippleTry/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"RippleTry/dense/Tensordot/concat_1?
RippleTry/dense/TensordotReshape*RippleTry/dense/Tensordot/MatMul:product:0+RippleTry/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
RippleTry/dense/Tensordot?
&RippleTry/dense/BiasAdd/ReadVariableOpReadVariableOp/rippletry_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&RippleTry/dense/BiasAdd/ReadVariableOp?
RippleTry/dense/BiasAddBiasAdd"RippleTry/dense/Tensordot:output:0.RippleTry/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
RippleTry/dense/BiasAdd?
RippleTry/dense/SigmoidSigmoid RippleTry/dense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
RippleTry/dense/Sigmoid?
IdentityIdentityRippleTry/dense/Sigmoid:y:07^RippleTry/batch_normalization/batchnorm/ReadVariableOp9^RippleTry/batch_normalization/batchnorm/ReadVariableOp_19^RippleTry/batch_normalization/batchnorm/ReadVariableOp_2;^RippleTry/batch_normalization/batchnorm/mul/ReadVariableOp'^RippleTry/dense/BiasAdd/ReadVariableOp)^RippleTry/dense/Tensordot/ReadVariableOp#^RippleTry/lstm/Read/ReadVariableOp%^RippleTry/lstm/Read_1/ReadVariableOp%^RippleTry/lstm/Read_2/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : 2p
6RippleTry/batch_normalization/batchnorm/ReadVariableOp6RippleTry/batch_normalization/batchnorm/ReadVariableOp2t
8RippleTry/batch_normalization/batchnorm/ReadVariableOp_18RippleTry/batch_normalization/batchnorm/ReadVariableOp_12t
8RippleTry/batch_normalization/batchnorm/ReadVariableOp_28RippleTry/batch_normalization/batchnorm/ReadVariableOp_22x
:RippleTry/batch_normalization/batchnorm/mul/ReadVariableOp:RippleTry/batch_normalization/batchnorm/mul/ReadVariableOp2P
&RippleTry/dense/BiasAdd/ReadVariableOp&RippleTry/dense/BiasAdd/ReadVariableOp2T
(RippleTry/dense/Tensordot/ReadVariableOp(RippleTry/dense/Tensordot/ReadVariableOp2H
"RippleTry/lstm/Read/ReadVariableOp"RippleTry/lstm/Read/ReadVariableOp2L
$RippleTry/lstm/Read_1/ReadVariableOp$RippleTry/lstm/Read_1/ReadVariableOp2L
$RippleTry/lstm/Read_2/ReadVariableOp$RippleTry/lstm/Read_2/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?A
?
 __inference_standard_lstm_692527

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????<2
MatMulj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????<2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
add^
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:?????????<2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :?????????:?????????: : :<:<:<* 
_read_only_resource_inputs
 *
bodyR
while_body_692442*
condR
while_cond_692441*_
output_shapesN
L: : : : :?????????:?????????: : :<:<:<*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:??????????????????:?????????:?????????:<:<:<*=
api_implements+)lstm_55cb023b-f9e0-4eff-9ca0-fbc632457c5d*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?A
?
 __inference_standard_lstm_696832

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????<2
MatMulj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????<2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
add^
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:?????????<2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :?????????:?????????: : :<:<:<* 
_read_only_resource_inputs
 *
bodyR
while_body_696747*
condR
while_cond_696746*_
output_shapesN
L: : : : :?????????:?????????: : :<:<:<*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:??????????????????:?????????:?????????:<:<:<*=
api_implements+)lstm_8d2ec9af-4917-4ad0-990a-8d3966677fc3*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?,
?
while_body_695280
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul_1?
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
	while/add?
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:?????????<2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:?????????2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:?????????2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:?????????2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :?????????:?????????: : :<:<:<: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:<:$	 

_output_shapes

:<: 


_output_shapes
:<
?H
?
)__inference_gpu_lstm_with_fallback_696491

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_1?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:??????????????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:??????????????????:?????????:?????????:<:<:<*=
api_implements+)lstm_10775f66-0d42-44f9-b303-bd1235de0720*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
??
?
;__inference___backward_gpu_lstm_with_fallback_694115_694291
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:?????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????25
3gradients/strided_slice_grad/StridedSliceGrad/begin?
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end?
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:?????????*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutation?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:?????????2&
$gradients/transpose_9_grad/transpose?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape?
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2"
 gradients/Squeeze_1_grad/Reshape?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:?????????2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:?????????2$
"gradients/transpose_grad/transpose?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod?
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_1_grad/Shape?
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_1?
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_2?
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_3?
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_4?
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_5?
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_6?
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_7?
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_8?
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_9?
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_10?
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_11?
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_12?
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_13?
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_14?
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_15?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:2
gradients/concat_1_grad/Slice?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_1?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_2?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_3?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_4?
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_5?
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_6?
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_7?
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_8?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_9?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_10?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_11?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_12?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_13?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_14?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_15?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
gradients/Reshape_grad/Shape?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:2 
gradients/Reshape_grad/Reshape?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_1_grad/Reshape?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_2_grad/Reshape?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_3_grad/Reshape?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_4_grad/Reshape?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_5_grad/Reshape?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_6_grad/Reshape?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_7_grad/Shape?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_7_grad/Reshape?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_8_grad/Reshape?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_9_grad/Reshape?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_10_grad/Reshape?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_11_grad/Reshape?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_12_grad/Reshape?
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_13_grad/Shape?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_13_grad/Reshape?
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_14_grad/Shape?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_14_grad/Reshape?
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_15_grad/Shape?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_15_grad/Reshape?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_1_grad/transpose?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_2_grad/transpose?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_3_grad/transpose?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_4_grad/transpose?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_5_grad/transpose?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_6_grad/transpose?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_7_grad/transpose?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_8_grad/transpose?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes
:x2
gradients/split_2_grad/concat?
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_grad/concat?
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rank?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape_1?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffset?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice?
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice_1~
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_2s

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_3u

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_4q

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:<2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????: :?????????::?????????:?????????::?????????:?????????:?????????:?::?????????:?????????: ::::::::: : : : *=
api_implements+)lstm_6f89c239-b136-43c3-957e-6d1798a93e4e*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_694290*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :1-
+
_output_shapes
:?????????: 

_output_shapes
::1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:	

_output_shapes
::1
-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?,
?
while_body_693935
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul_1?
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
	while/add?
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:?????????<2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:?????????2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:?????????2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:?????????2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :?????????:?????????: : :<:<:<: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:<:$	 

_output_shapes

:<: 


_output_shapes
:<
?,
?
while_body_694609
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul_1?
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
	while/add?
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:?????????<2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:?????????2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:?????????2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:?????????2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :?????????:?????????: : :<:<:<: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:<:$	 

_output_shapes

:<: 


_output_shapes
:<
??
?
;__inference___backward_gpu_lstm_with_fallback_692622_692798
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0?
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :??????????????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????25
3gradients/strided_slice_grad/StridedSliceGrad/begin?
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end?
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :??????????????????*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutation?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :??????????????????2&
$gradients/transpose_9_grad/transpose?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape?
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2"
 gradients/Squeeze_1_grad/Reshape?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :??????????????????2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:??????????????????:?????????:?????????:?2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :??????????????????2$
"gradients/transpose_grad/transpose?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod?
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_1_grad/Shape?
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_1?
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_2?
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_3?
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_4?
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_5?
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_6?
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_7?
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_8?
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_9?
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_10?
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_11?
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_12?
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_13?
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_14?
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_15?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:2
gradients/concat_1_grad/Slice?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_1?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_2?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_3?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_4?
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_5?
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_6?
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_7?
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_8?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_9?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_10?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_11?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_12?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_13?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_14?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_15?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
gradients/Reshape_grad/Shape?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:2 
gradients/Reshape_grad/Reshape?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_1_grad/Reshape?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_2_grad/Reshape?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_3_grad/Reshape?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_4_grad/Reshape?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_5_grad/Reshape?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_6_grad/Reshape?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_7_grad/Shape?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_7_grad/Reshape?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_8_grad/Reshape?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_9_grad/Reshape?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_10_grad/Reshape?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_11_grad/Reshape?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_12_grad/Reshape?
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_13_grad/Shape?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_13_grad/Reshape?
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_14_grad/Shape?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_14_grad/Reshape?
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_15_grad/Shape?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_15_grad/Reshape?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_1_grad/transpose?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_2_grad/transpose?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_3_grad/transpose?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_4_grad/transpose?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_5_grad/transpose?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_6_grad/transpose?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_7_grad/transpose?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_8_grad/transpose?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes
:x2
gradients/split_2_grad/concat?
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_grad/concat?
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rank?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape_1?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffset?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice?
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice_1?
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_2s

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_3u

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_4q

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:<2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:??????????????????:?????????:?????????: :??????????????????::?????????:?????????::??????????????????:?????????:?????????:?::?????????:?????????: ::::::::: : : : *=
api_implements+)lstm_55cb023b-f9e0-4eff-9ca0-fbc632457c5d*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_692797*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????::6
4
_output_shapes"
 :??????????????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: ::6
4
_output_shapes"
 :??????????????????: 

_output_shapes
::1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:	

_output_shapes
:::
6
4
_output_shapes"
 :??????????????????:1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?#
?
@__inference_lstm_layer_call_and_return_conditional_losses_692343

inputs.
read_readvariableop_resource:<0
read_1_readvariableop_resource:<,
read_2_readvariableop_resource:<

identity_3??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1?
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:<*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:<2

Identity?
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:<*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:<2

Identity_1?
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:<*
dtype02
Read_2/ReadVariableOph

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:<2

Identity_2?
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:?????????:??????????????????:?????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference_standard_lstm_6920702
PartitionedCall?

Identity_3IdentityPartitionedCall:output:1^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity_3"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
%__inference_lstm_layer_call_fn_696202
inputs_0
unknown:<
	unknown_0:<
	unknown_1:<
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_6923432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?,
?
while_body_691985
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul_1?
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
	while/add?
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:?????????<2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:?????????2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:?????????2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:?????????2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :?????????:?????????: : :<:<:<: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:<:$	 

_output_shapes

:<: 


_output_shapes
:<
?H
?
)__inference_gpu_lstm_with_fallback_696926

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_1?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:??????????????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:??????????????????:?????????:?????????:<:<:<*=
api_implements+)lstm_8d2ec9af-4917-4ad0-990a-8d3966677fc3*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?T
?
'__forward_gpu_lstm_with_fallback_696667

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:??????????????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:??????????????????:?????????:?????????:<:<:<*=
api_implements+)lstm_10775f66-0d42-44f9-b303-bd1235de0720*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_696492_696668*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
??
?
;__inference___backward_gpu_lstm_with_fallback_696492_696668
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0?
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :??????????????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????25
3gradients/strided_slice_grad/StridedSliceGrad/begin?
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end?
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :??????????????????*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutation?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :??????????????????2&
$gradients/transpose_9_grad/transpose?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape?
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2"
 gradients/Squeeze_1_grad/Reshape?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :??????????????????2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:??????????????????:?????????:?????????:?2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :??????????????????2$
"gradients/transpose_grad/transpose?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod?
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_1_grad/Shape?
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_1?
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_2?
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_3?
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_4?
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_5?
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_6?
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_7?
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_8?
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_9?
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_10?
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_11?
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_12?
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_13?
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_14?
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_15?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:2
gradients/concat_1_grad/Slice?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_1?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_2?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_3?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_4?
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_5?
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_6?
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_7?
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_8?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_9?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_10?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_11?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_12?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_13?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_14?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_15?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
gradients/Reshape_grad/Shape?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:2 
gradients/Reshape_grad/Reshape?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_1_grad/Reshape?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_2_grad/Reshape?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_3_grad/Reshape?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_4_grad/Reshape?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_5_grad/Reshape?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_6_grad/Reshape?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_7_grad/Shape?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_7_grad/Reshape?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_8_grad/Reshape?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_9_grad/Reshape?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_10_grad/Reshape?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_11_grad/Reshape?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_12_grad/Reshape?
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_13_grad/Shape?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_13_grad/Reshape?
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_14_grad/Shape?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_14_grad/Reshape?
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_15_grad/Shape?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_15_grad/Reshape?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_1_grad/transpose?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_2_grad/transpose?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_3_grad/transpose?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_4_grad/transpose?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_5_grad/transpose?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_6_grad/transpose?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_7_grad/transpose?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_8_grad/transpose?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes
:x2
gradients/split_2_grad/concat?
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_grad/concat?
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rank?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape_1?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffset?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice?
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice_1?
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_2s

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_3u

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_4q

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:<2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:??????????????????:?????????:?????????: :??????????????????::?????????:?????????::??????????????????:?????????:?????????:?::?????????:?????????: ::::::::: : : : *=
api_implements+)lstm_10775f66-0d42-44f9-b303-bd1235de0720*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_696667*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????::6
4
_output_shapes"
 :??????????????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: ::6
4
_output_shapes"
 :??????????????????: 

_output_shapes
::1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:	

_output_shapes
:::
6
4
_output_shapes"
 :??????????????????:1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
4__inference_batch_normalization_layer_call_fn_698015

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_6937142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?"
?
@__inference_lstm_layer_call_and_return_conditional_losses_694967

inputs.
read_readvariableop_resource:<0
read_1_readvariableop_resource:<,
read_2_readvariableop_resource:<

identity_3??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1?
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:<*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:<2

Identity?
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:<*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:<2

Identity_1?
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:<*
dtype02
Read_2/ReadVariableOph

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:<2

Identity_2?
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:?????????:?????????:?????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference_standard_lstm_6946942
PartitionedCall?

Identity_3IdentityPartitionedCall:output:1^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_3"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_layer_call_fn_698028

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_6937742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?#
?
@__inference_lstm_layer_call_and_return_conditional_losses_692800

inputs.
read_readvariableop_resource:<0
read_1_readvariableop_resource:<,
read_2_readvariableop_resource:<

identity_3??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1?
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:<*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:<2

Identity?
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:<*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:<2

Identity_1?
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:<*
dtype02
Read_2/ReadVariableOph

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:<2

Identity_2?
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:?????????:??????????????????:?????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference_standard_lstm_6925272
PartitionedCall?

Identity_3IdentityPartitionedCall:output:1^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity_3"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?@
?
 __inference_standard_lstm_695365

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????<2
MatMulj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????<2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
add^
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:?????????<2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :?????????:?????????: : :<:<:<* 
_read_only_resource_inputs
 *
bodyR
while_body_695280*
condR
while_cond_695279*_
output_shapesN
L: : : : :?????????:?????????: : :<:<:<*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:?????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_333a86cd-1619-47ae-aeec-a18c90718136*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?A
?
 __inference_standard_lstm_692070

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:?????????<2
MatMulj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:?????????<2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
add^
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:?????????<2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*`
_output_shapesN
L: : : : :?????????:?????????: : :<:<:<* 
_read_only_resource_inputs
 *
bodyR
while_body_691985*
condR
while_cond_691984*_
output_shapesN
L: : : : :?????????:?????????: : :<:<:<*
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:?????????2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:??????????????????:?????????:?????????:<:<:<*=
api_implements+)lstm_84ff0d2d-5e3a-4329-93ff-605791529804*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?
?
%__inference_lstm_layer_call_fn_696235

inputs
unknown:<
	unknown_0:<
	unknown_1:<
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_6949672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
while_cond_697181
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_697181___redundant_placeholder04
0while_while_cond_697181___redundant_placeholder14
0while_while_cond_697181___redundant_placeholder24
0while_while_cond_697181___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_698189

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0*
seed???)*
seed22&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
C__inference_dropout_layer_call_and_return_conditional_losses_694516

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0*
seed???)*
seed22&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_693714

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?T
?
'__forward_gpu_lstm_with_fallback_692340

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:??????????????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:??????????????????:?????????:?????????:<:<:<*=
api_implements+)lstm_84ff0d2d-5e3a-4329-93ff-605791529804*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_692165_692341*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?	
?
*__inference_RippleTry_layer_call_fn_694403
input_1
unknown:<
	unknown_0:<
	unknown_1:<
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_RippleTry_layer_call_and_return_conditional_losses_6943822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?H
?
)__inference_gpu_lstm_with_fallback_695939

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_1?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:?????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:?????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_3dcebcae-8c80-4c1d-962a-8718205d4aa8*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?H
?
)__inference_gpu_lstm_with_fallback_691680

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_1?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:?????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:?????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_1109ec47-c15b-4e26-9c01-f618535fb6ce*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?S
?
'__forward_gpu_lstm_with_fallback_697972

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:?????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:?????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_d31ccf94-a07e-4964-8059-939fe35bedbb*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_697797_697973*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_697990

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
(__inference_dropout_layer_call_fn_697985

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_6945162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?*
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_693774

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_layer_call_fn_698041

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_6943272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?S
?
'__forward_gpu_lstm_with_fallback_694290

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:?????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:?????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_6f89c239-b136-43c3-957e-6d1798a93e4e*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_694115_694291*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?
D
(__inference_dropout_layer_call_fn_697980

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_6943062
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
?
A__inference_dense_layer_call_and_return_conditional_losses_694375

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdde
SigmoidSigmoidBiasAdd:output:0*
T0*+
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?S
?
'__forward_gpu_lstm_with_fallback_697537

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1?
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:?????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:?????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_5796cf19-f097-407f-ba62-351741090953*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_697362_697538*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
? 
?
A__inference_dense_layer_call_and_return_conditional_losses_698229

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdde
SigmoidSigmoidBiasAdd:output:0*
T0*+
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_1_layer_call_fn_698167

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_6943422
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?#
?
@__inference_lstm_layer_call_and_return_conditional_losses_696670
inputs_0.
read_readvariableop_resource:<0
read_1_readvariableop_resource:<,
read_2_readvariableop_resource:<

identity_3??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1?
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:<*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:<2

Identity?
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:<*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:<2

Identity_1?
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:<*
dtype02
Read_2/ReadVariableOph

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:<2

Identity_2?
PartitionedCallPartitionedCallinputs_0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:?????????:??????????????????:?????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference_standard_lstm_6963972
PartitionedCall?

Identity_3IdentityPartitionedCall:output:1^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity_3"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?H
?
)__inference_gpu_lstm_with_fallback_695459

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_1?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:?????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:?????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_333a86cd-1619-47ae-aeec-a18c90718136*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
??
?
;__inference___backward_gpu_lstm_with_fallback_691681_691857
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:?????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????25
3gradients/strided_slice_grad/StridedSliceGrad/begin?
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end?
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:?????????*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutation?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:?????????2&
$gradients/transpose_9_grad/transpose?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape?
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2"
 gradients/Squeeze_1_grad/Reshape?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:?????????2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:?????????2$
"gradients/transpose_grad/transpose?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod?
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_1_grad/Shape?
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_1?
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_2?
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_3?
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_4?
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_5?
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_6?
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_7?
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_8?
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_9?
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_10?
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_11?
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_12?
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_13?
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_14?
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_15?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:2
gradients/concat_1_grad/Slice?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_1?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_2?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_3?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_4?
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_5?
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_6?
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_7?
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_8?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_9?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_10?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_11?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_12?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_13?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_14?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_15?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
gradients/Reshape_grad/Shape?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:2 
gradients/Reshape_grad/Reshape?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_1_grad/Reshape?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_2_grad/Reshape?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_3_grad/Reshape?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_4_grad/Reshape?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_5_grad/Reshape?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_6_grad/Reshape?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_7_grad/Shape?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_7_grad/Reshape?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_8_grad/Reshape?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_9_grad/Reshape?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_10_grad/Reshape?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_11_grad/Reshape?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_12_grad/Reshape?
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_13_grad/Shape?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_13_grad/Reshape?
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_14_grad/Shape?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_14_grad/Reshape?
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_15_grad/Shape?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_15_grad/Reshape?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_1_grad/transpose?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_2_grad/transpose?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_3_grad/transpose?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_4_grad/transpose?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_5_grad/transpose?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_6_grad/transpose?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_7_grad/transpose?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_8_grad/transpose?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes
:x2
gradients/split_2_grad/concat?
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_grad/concat?
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rank?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape_1?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffset?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice?
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice_1~
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_2s

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_3u

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_4q

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:<2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????: :?????????::?????????:?????????::?????????:?????????:?????????:?::?????????:?????????: ::::::::: : : : *=
api_implements+)lstm_1109ec47-c15b-4e26-9c01-f618535fb6ce*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_691856*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :1-
+
_output_shapes
:?????????: 

_output_shapes
::1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:	

_output_shapes
::1
-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?,
?
while_body_697182
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul_1?
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
	while/add?
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:?????????<2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:?????????2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:?????????2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:?????????2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :?????????:?????????: : :<:<:<: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:<:$	 

_output_shapes

:<: 


_output_shapes
:<
?,
?
while_body_691501
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul?
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:?????????<2
while/MatMul_1?
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:?????????<2
	while/add?
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:?????????<2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:?????????2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:?????????2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:?????????2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:?????????2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:?????????2
while/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :?????????:?????????: : :<:<:<: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:<:$	 

_output_shapes

:<: 


_output_shapes
:<
??
?
;__inference___backward_gpu_lstm_with_fallback_697797_697973
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5?u
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:?????????2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:?????????2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4?
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape?
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
?????????25
3gradients/strided_slice_grad/StridedSliceGrad/begin?
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end?
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:?????????*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutation?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:?????????2&
$gradients/transpose_9_grad/transpose?
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape?
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2 
gradients/Squeeze_grad/Reshape?
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape?
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:?????????2"
 gradients/Squeeze_1_grad/Reshape?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:?????????2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:?????????2$
"gradients/transpose_grad/transpose?
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shape?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2#
!gradients/ExpandDims_grad/Reshape?
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod?
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_1_grad/Shape?
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_1?
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_2?
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_3?
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_4?
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_5?
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_6?
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:?2!
gradients/concat_1_grad/Shape_7?
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_8?
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2!
gradients/concat_1_grad/Shape_9?
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_10?
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_11?
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_12?
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_13?
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_14?
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:2"
 gradients/concat_1_grad/Shape_15?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:2
gradients/concat_1_grad/Slice?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_1?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_2?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_3?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_4?
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_5?
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_6?
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:?2!
gradients/concat_1_grad/Slice_7?
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_8?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:2!
gradients/concat_1_grad/Slice_9?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_10?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_11?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_12?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_13?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_14?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:2"
 gradients/concat_1_grad/Slice_15?
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
gradients/Reshape_grad/Shape?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:2 
gradients/Reshape_grad/Reshape?
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_1_grad/Shape?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_1_grad/Reshape?
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_2_grad/Shape?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_2_grad/Reshape?
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_3_grad/Shape?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_3_grad/Reshape?
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/Shape?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_4_grad/Reshape?
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/Shape?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_5_grad/Reshape?
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/Shape?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_6_grad/Reshape?
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_7_grad/Shape?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:2"
 gradients/Reshape_7_grad/Reshape?
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/Shape?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_8_grad/Reshape?
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/Shape?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:2"
 gradients/Reshape_9_grad/Reshape?
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/Shape?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_10_grad/Reshape?
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/Shape?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_11_grad/Reshape?
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/Shape?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_12_grad/Reshape?
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_13_grad/Shape?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_13_grad/Reshape?
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_14_grad/Shape?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_14_grad/Reshape?
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_15_grad/Shape?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:2#
!gradients/Reshape_15_grad/Reshape?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_1_grad/transpose?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_2_grad/transpose?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_3_grad/transpose?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_4_grad/transpose?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_5_grad/transpose?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_6_grad/transpose?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_7_grad/transpose?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2&
$gradients/transpose_8_grad/transpose?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes
:x2
gradients/split_2_grad/concat?
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_grad/concat?
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:<2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rank?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod?
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape?
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:<2
gradients/concat_grad/Shape_1?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffset?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice?
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:<2
gradients/concat_grad/Slice_1~
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:?????????2

Identity_2s

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_3u

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:<2

Identity_4q

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:<2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????: :?????????::?????????:?????????::?????????:?????????:?????????:?::?????????:?????????: ::::::::: : : : *=
api_implements+)lstm_d31ccf94-a07e-4964-8059-939fe35bedbb*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_697972*
go_backwards( *

time_major( :- )
'
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :1-
+
_output_shapes
:?????????: 

_output_shapes
::1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:	

_output_shapes
::1
-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:1-
+
_output_shapes
:?????????:!

_output_shapes	
:?: 

_output_shapes
::-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
while_cond_697616
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_697616___redundant_placeholder04
0while_while_cond_697616___redundant_placeholder14
0while_while_cond_697616___redundant_placeholder24
0while_while_cond_697616___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
?#
?
@__inference_lstm_layer_call_and_return_conditional_losses_697105
inputs_0.
read_readvariableop_resource:<0
read_1_readvariableop_resource:<,
read_2_readvariableop_resource:<

identity_3??Read/ReadVariableOp?Read_1/ReadVariableOp?Read_2/ReadVariableOpF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1?
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:<*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:<2

Identity?
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:<*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:<2

Identity_1?
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:<*
dtype02
Read_2/ReadVariableOph

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:<2

Identity_2?
PartitionedCallPartitionedCallinputs_0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:?????????:??????????????????:?????????:?????????: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference_standard_lstm_6968322
PartitionedCall?

Identity_3IdentityPartitionedCall:output:1^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity_3"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?H
?
)__inference_gpu_lstm_with_fallback_692164

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_1?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:??????????????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityt

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:??????????????????:?????????:?????????:<:<:<*=
api_implements+)lstm_84ff0d2d-5e3a-4329-93ff-605791529804*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?F
?
__inference__traced_save_698348
file_prefix8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop4
0savev2_lstm_lstm_cell_kernel_read_readvariableop>
:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop2
.savev2_lstm_lstm_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop;
7savev2_adam_lstm_lstm_cell_kernel_m_read_readvariableopE
Asavev2_adam_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop9
5savev2_adam_lstm_lstm_cell_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop;
7savev2_adam_lstm_lstm_cell_kernel_v_read_readvariableopE
Asavev2_adam_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop9
5savev2_adam_lstm_lstm_cell_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop0savev2_lstm_lstm_cell_kernel_read_readvariableop:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop.savev2_lstm_lstm_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop7savev2_adam_lstm_lstm_cell_kernel_m_read_readvariableopAsavev2_adam_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop5savev2_adam_lstm_lstm_cell_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop7savev2_adam_lstm_lstm_cell_kernel_v_read_readvariableopAsavev2_adam_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop5savev2_adam_lstm_lstm_cell_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 */
dtypes%
#2!	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::: : : : : :<:<:<: : : : :::::<:<:<:::::<:<:<: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:<:$ 

_output_shapes

:<: 

_output_shapes
:<:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:<:$ 

_output_shapes

:<: 

_output_shapes
:<: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:<:$ 

_output_shapes

:<:  

_output_shapes
:<:!

_output_shapes
: 
?H
?
)__inference_gpu_lstm_with_fallback_694788

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4?u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(::::*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split2	
split_1e

zeros_likeConst*
_output_shapes
:<*
dtype0*
valueB<*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis{
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes
:x2
concath
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1c
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes
:2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2g
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3g
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4g
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:2
transpose_5h
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:2
transpose_6h
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:2
transpose_7h
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:2
transpose_8h
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:?2
	Reshape_7h
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:2
	Reshape_8h
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:2
	Reshape_9j

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:2

Reshape_10j

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:2

Reshape_11j

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:2

Reshape_12j

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:2

Reshape_13j

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:2

Reshape_14j

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_1?
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:?????????:?????????:?????????:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm?
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimej
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:?????????2

Identityk

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:?????????2

Identity_1h

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity_2j

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????:?????????:?????????:<:<:<*=
api_implements+)lstm_21c69771-b438-47fb-9c6b-4c500860443c*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_h:OK
'
_output_shapes
:?????????
 
_user_specified_nameinit_c:FB

_output_shapes

:<
 
_user_specified_namekernel:PL

_output_shapes

:<
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:<

_user_specified_namebias
?
b
C__inference_dropout_layer_call_and_return_conditional_losses_698002

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0*
seed???)*
seed22&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_698454
file_prefix8
*assignvariableop_batch_normalization_gamma:9
+assignvariableop_1_batch_normalization_beta:@
2assignvariableop_2_batch_normalization_moving_mean:D
6assignvariableop_3_batch_normalization_moving_variance:1
assignvariableop_4_dense_kernel:+
assignvariableop_5_dense_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: ;
)assignvariableop_11_lstm_lstm_cell_kernel:<E
3assignvariableop_12_lstm_lstm_cell_recurrent_kernel:<5
'assignvariableop_13_lstm_lstm_cell_bias:<#
assignvariableop_14_total: #
assignvariableop_15_count: %
assignvariableop_16_total_1: %
assignvariableop_17_count_1: B
4assignvariableop_18_adam_batch_normalization_gamma_m:A
3assignvariableop_19_adam_batch_normalization_beta_m:9
'assignvariableop_20_adam_dense_kernel_m:3
%assignvariableop_21_adam_dense_bias_m:B
0assignvariableop_22_adam_lstm_lstm_cell_kernel_m:<L
:assignvariableop_23_adam_lstm_lstm_cell_recurrent_kernel_m:<<
.assignvariableop_24_adam_lstm_lstm_cell_bias_m:<B
4assignvariableop_25_adam_batch_normalization_gamma_v:A
3assignvariableop_26_adam_batch_normalization_beta_v:9
'assignvariableop_27_adam_dense_kernel_v:3
%assignvariableop_28_adam_dense_bias_v:B
0assignvariableop_29_adam_lstm_lstm_cell_kernel_v:<L
:assignvariableop_30_adam_lstm_lstm_cell_recurrent_kernel_v:<<
.assignvariableop_31_adam_lstm_lstm_cell_bias_v:<
identity_33??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::*/
dtypes%
#2!	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp*assignvariableop_batch_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp+assignvariableop_1_batch_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp2assignvariableop_2_batch_normalization_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp6assignvariableop_3_batch_normalization_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp)assignvariableop_11_lstm_lstm_cell_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp3assignvariableop_12_lstm_lstm_cell_recurrent_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp'assignvariableop_13_lstm_lstm_cell_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp4assignvariableop_18_adam_batch_normalization_gamma_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp3assignvariableop_19_adam_batch_normalization_beta_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp%assignvariableop_21_adam_dense_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp0assignvariableop_22_adam_lstm_lstm_cell_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp:assignvariableop_23_adam_lstm_lstm_cell_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp.assignvariableop_24_adam_lstm_lstm_cell_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp4assignvariableop_25_adam_batch_normalization_gamma_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp3assignvariableop_26_adam_batch_normalization_beta_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_dense_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp0assignvariableop_29_adam_lstm_lstm_cell_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp:assignvariableop_30_adam_lstm_lstm_cell_recurrent_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp.assignvariableop_31_adam_lstm_lstm_cell_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_319
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_32Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_32?
Identity_33IdentityIdentity_32:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_33"#
identity_33Identity_33:output:0*U
_input_shapesD
B: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?*
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_694485

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input_14
serving_default_input_1:0?????????=
dense4
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?;
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
	optimizer
	variables
	regularization_losses

trainable_variables
	keras_api

signatures
t__call__
*u&call_and_return_all_conditional_losses
v_default_save_signature"?8
_tf_keras_network?8{"name": "RippleTry", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "RippleTry", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 15, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": 0}, "shared_object_id": 1}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": 1}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": 2}, "name": "dropout", "inbound_nodes": [[["lstm", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": 3}, "name": "dropout_1", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": 13}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}, "shared_object_id": 16, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 16, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 16, 1]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "RippleTry", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 15, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": 0}, "shared_object_id": 1}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": 1}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": 2}, "name": "dropout", "inbound_nodes": [[["lstm", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 10}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": 3}, "name": "dropout_1", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": 13}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 15}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}, "shared_object_id": 18}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.004999999888241291, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
w__call__
*x&call_and_return_all_conditional_losses"?
_tf_keras_rnn_layer?{"name": "lstm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 15, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": 0}, "shared_object_id": 1}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": 1}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 5, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 19}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 1]}}
?
	variables
regularization_losses
trainable_variables
	keras_api
y__call__
*z&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": 2}, "inbound_nodes": [[["lstm", 0, 0, {}]]], "shared_object_id": 6}
?

axis
	gamma
beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
{__call__
*|&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 10}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 15}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 15]}}
?
 	variables
!regularization_losses
"trainable_variables
#	keras_api
}__call__
*~&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": 3}, "inbound_nodes": [[["batch_normalization", 0, 0, {}]]], "shared_object_id": 12}
?

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": 13}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 15}}, "shared_object_id": 21}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 15]}}
?
*iter

+beta_1

,beta_2
	-decay
.learning_ratemfmg$mh%mi/mj0mk1mlvmvn$vo%vp/vq0vr1vs"
	optimizer
_
/0
01
12
3
4
5
6
$7
%8"
trackable_list_wrapper
 "
trackable_list_wrapper
Q
/0
01
12
3
4
$5
%6"
trackable_list_wrapper
?
2layer_regularization_losses
	variables
3non_trainable_variables
4layer_metrics
	regularization_losses
5metrics

trainable_variables

6layers
t__call__
v_default_save_signature
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?	
7
state_size

/kernel
0recurrent_kernel
1bias
8	variables
9regularization_losses
:trainable_variables
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LSTMCell", "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 15, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": 0}, "shared_object_id": 1}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": 1}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 4}
 "
trackable_list_wrapper
5
/0
01
12"
trackable_list_wrapper
 "
trackable_list_wrapper
5
/0
01
12"
trackable_list_wrapper
?
<layer_regularization_losses
	variables
=non_trainable_variables

>states
?layer_metrics
regularization_losses
@metrics
trainable_variables

Alayers
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Blayer_regularization_losses
Cnon_trainable_variables
	variables
Dlayer_metrics
Emetrics
regularization_losses

Flayers
trainable_variables
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Glayer_regularization_losses
Hnon_trainable_variables
	variables
Ilayer_metrics
Jmetrics
regularization_losses

Klayers
trainable_variables
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Llayer_regularization_losses
Mnon_trainable_variables
 	variables
Nlayer_metrics
Ometrics
!regularization_losses

Players
"trainable_variables
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
:2dense/kernel
:2
dense/bias
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
?
Qlayer_regularization_losses
Rnon_trainable_variables
&	variables
Slayer_metrics
Tmetrics
'regularization_losses

Ulayers
(trainable_variables
__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
':%<2lstm/lstm_cell/kernel
1:/<2lstm/lstm_cell/recurrent_kernel
!:<2lstm/lstm_cell/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
V0
W1"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
5
/0
01
12"
trackable_list_wrapper
 "
trackable_list_wrapper
5
/0
01
12"
trackable_list_wrapper
?
Xlayer_regularization_losses
Ynon_trainable_variables
8	variables
Zlayer_metrics
[metrics
9regularization_losses

\layers
:trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	]total
	^count
_	variables
`	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 22}
?
	atotal
	bcount
c
_fn_kwargs
d	variables
e	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}, "shared_object_id": 18}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
]0
^1"
trackable_list_wrapper
-
_	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
a0
b1"
trackable_list_wrapper
-
d	variables"
_generic_user_object
,:*2 Adam/batch_normalization/gamma/m
+:)2Adam/batch_normalization/beta/m
#:!2Adam/dense/kernel/m
:2Adam/dense/bias/m
,:*<2Adam/lstm/lstm_cell/kernel/m
6:4<2&Adam/lstm/lstm_cell/recurrent_kernel/m
&:$<2Adam/lstm/lstm_cell/bias/m
,:*2 Adam/batch_normalization/gamma/v
+:)2Adam/batch_normalization/beta/v
#:!2Adam/dense/kernel/v
:2Adam/dense/bias/v
,:*<2Adam/lstm/lstm_cell/kernel/v
6:4<2&Adam/lstm/lstm_cell/recurrent_kernel/v
&:$<2Adam/lstm/lstm_cell/bias/v
?2?
*__inference_RippleTry_layer_call_fn_694403
*__inference_RippleTry_layer_call_fn_695180
*__inference_RippleTry_layer_call_fn_695203
*__inference_RippleTry_layer_call_fn_695072?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_RippleTry_layer_call_and_return_conditional_losses_695683
E__inference_RippleTry_layer_call_and_return_conditional_losses_696191
E__inference_RippleTry_layer_call_and_return_conditional_losses_695099
E__inference_RippleTry_layer_call_and_return_conditional_losses_695126?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_691904?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"
input_1?????????
?2?
%__inference_lstm_layer_call_fn_696202
%__inference_lstm_layer_call_fn_696213
%__inference_lstm_layer_call_fn_696224
%__inference_lstm_layer_call_fn_696235?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_lstm_layer_call_and_return_conditional_losses_696670
@__inference_lstm_layer_call_and_return_conditional_losses_697105
@__inference_lstm_layer_call_and_return_conditional_losses_697540
@__inference_lstm_layer_call_and_return_conditional_losses_697975?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dropout_layer_call_fn_697980
(__inference_dropout_layer_call_fn_697985?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dropout_layer_call_and_return_conditional_losses_697990
C__inference_dropout_layer_call_and_return_conditional_losses_698002?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_batch_normalization_layer_call_fn_698015
4__inference_batch_normalization_layer_call_fn_698028
4__inference_batch_normalization_layer_call_fn_698041
4__inference_batch_normalization_layer_call_fn_698054?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_698074
O__inference_batch_normalization_layer_call_and_return_conditional_losses_698108
O__inference_batch_normalization_layer_call_and_return_conditional_losses_698128
O__inference_batch_normalization_layer_call_and_return_conditional_losses_698162?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_1_layer_call_fn_698167
*__inference_dropout_1_layer_call_fn_698172?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_1_layer_call_and_return_conditional_losses_698177
E__inference_dropout_1_layer_call_and_return_conditional_losses_698189?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_dense_layer_call_fn_698198?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_layer_call_and_return_conditional_losses_698229?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_695157input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
E__inference_RippleTry_layer_call_and_return_conditional_losses_695099t	/01$%<?9
2?/
%?"
input_1?????????
p 

 
? ")?&
?
0?????????
? ?
E__inference_RippleTry_layer_call_and_return_conditional_losses_695126t	/01$%<?9
2?/
%?"
input_1?????????
p

 
? ")?&
?
0?????????
? ?
E__inference_RippleTry_layer_call_and_return_conditional_losses_695683s	/01$%;?8
1?.
$?!
inputs?????????
p 

 
? ")?&
?
0?????????
? ?
E__inference_RippleTry_layer_call_and_return_conditional_losses_696191s	/01$%;?8
1?.
$?!
inputs?????????
p

 
? ")?&
?
0?????????
? ?
*__inference_RippleTry_layer_call_fn_694403g	/01$%<?9
2?/
%?"
input_1?????????
p 

 
? "???????????
*__inference_RippleTry_layer_call_fn_695072g	/01$%<?9
2?/
%?"
input_1?????????
p

 
? "???????????
*__inference_RippleTry_layer_call_fn_695180f	/01$%;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
*__inference_RippleTry_layer_call_fn_695203f	/01$%;?8
1?.
$?!
inputs?????????
p

 
? "???????????
!__inference__wrapped_model_691904t	/01$%4?1
*?'
%?"
input_1?????????
? "1?.
,
dense#? 
dense??????????
O__inference_batch_normalization_layer_call_and_return_conditional_losses_698074|@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_698108|@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_698128j7?4
-?*
$?!
inputs?????????
p 
? ")?&
?
0?????????
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_698162j7?4
-?*
$?!
inputs?????????
p
? ")?&
?
0?????????
? ?
4__inference_batch_normalization_layer_call_fn_698015o@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
4__inference_batch_normalization_layer_call_fn_698028o@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
4__inference_batch_normalization_layer_call_fn_698041]7?4
-?*
$?!
inputs?????????
p 
? "???????????
4__inference_batch_normalization_layer_call_fn_698054]7?4
-?*
$?!
inputs?????????
p
? "???????????
A__inference_dense_layer_call_and_return_conditional_losses_698229d$%3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
&__inference_dense_layer_call_fn_698198W$%3?0
)?&
$?!
inputs?????????
? "???????????
E__inference_dropout_1_layer_call_and_return_conditional_losses_698177d7?4
-?*
$?!
inputs?????????
p 
? ")?&
?
0?????????
? ?
E__inference_dropout_1_layer_call_and_return_conditional_losses_698189d7?4
-?*
$?!
inputs?????????
p
? ")?&
?
0?????????
? ?
*__inference_dropout_1_layer_call_fn_698167W7?4
-?*
$?!
inputs?????????
p 
? "???????????
*__inference_dropout_1_layer_call_fn_698172W7?4
-?*
$?!
inputs?????????
p
? "???????????
C__inference_dropout_layer_call_and_return_conditional_losses_697990d7?4
-?*
$?!
inputs?????????
p 
? ")?&
?
0?????????
? ?
C__inference_dropout_layer_call_and_return_conditional_losses_698002d7?4
-?*
$?!
inputs?????????
p
? ")?&
?
0?????????
? ?
(__inference_dropout_layer_call_fn_697980W7?4
-?*
$?!
inputs?????????
p 
? "???????????
(__inference_dropout_layer_call_fn_697985W7?4
-?*
$?!
inputs?????????
p
? "???????????
@__inference_lstm_layer_call_and_return_conditional_losses_696670?/01O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "2?/
(?%
0??????????????????
? ?
@__inference_lstm_layer_call_and_return_conditional_losses_697105?/01O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "2?/
(?%
0??????????????????
? ?
@__inference_lstm_layer_call_and_return_conditional_losses_697540q/01??<
5?2
$?!
inputs?????????

 
p 

 
? ")?&
?
0?????????
? ?
@__inference_lstm_layer_call_and_return_conditional_losses_697975q/01??<
5?2
$?!
inputs?????????

 
p

 
? ")?&
?
0?????????
? ?
%__inference_lstm_layer_call_fn_696202}/01O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"???????????????????
%__inference_lstm_layer_call_fn_696213}/01O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"???????????????????
%__inference_lstm_layer_call_fn_696224d/01??<
5?2
$?!
inputs?????????

 
p 

 
? "???????????
%__inference_lstm_layer_call_fn_696235d/01??<
5?2
$?!
inputs?????????

 
p

 
? "???????????
$__inference_signature_wrapper_695157	/01$%??<
? 
5?2
0
input_1%?"
input_1?????????"1?.
,
dense#? 
dense?????????