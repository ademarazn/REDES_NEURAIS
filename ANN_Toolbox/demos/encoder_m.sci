// Tight 4-2-4 encoder
// on a backpropagation ANN with biases and momentum

rand('seed',0);

// network def.
//  - neurons per layer, including input
N  = [4,2,4];

// inputs
x = [1,0,0,0;
     0,1,0,0;
     0,0,1,0;
     0,0,0,1]';

// targets, at training stage is acts as identity network
t = x;

// learning parameter
lp = [2.5,0,0.9,0.25];

// init randomize weights between:
r = [-1,7];

W = ann_FF_init(N,r);
Delta_W_old = hypermat(size(W)');

// 200 epochs are enough to ilustrate
T = 200;
[W,Delta_W_old] = ann_FF_Mom_online(x,t,N,W,lp,T,Delta_W_old);

// full run
ann_FF_run(x,N,W)

// encoder
encoder = ann_FF_run(x,N,W,[2,2])
// decoder
decoder = ann_FF_run(encoder,N,W,[3,3])

