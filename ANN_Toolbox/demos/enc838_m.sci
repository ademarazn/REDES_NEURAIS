// Tight 8-3-8 encoder
// on a backpropagation ANN with biases and momentum

rand('seed',0);

// network def.
//  - neurons per layer, including input
N  = [8,3,8];

// inputs
x = [1,0,0,0,0,0,0,0;
     0,1,0,0,0,0,0,0;
     0,0,1,0,0,0,0,0;
     0,0,0,1,0,0,0,0;
     0,0,0,0,1,0,0,0;
     0,0,0,0,0,1,0,0;
     0,0,0,0,0,0,1,0;
     0,0,0,0,0,0,0,1]';

// targets, at training stage is acts as identity network
t = x;

// learning parameter
lp = [1.5, 0.07, 0.8, 0.1];

// init randomize weights between:
r = [-10,15];
rb = r;

W = ann_FF_init(N,r,rb);
Delta_W_old = hypermat(size(W)');

// 500 epochs are enough to ilustrate
T = 500;
[W,Delta_W_old] = ann_FF_Mom_online(x,t,N,W,lp,T,Delta_W_old);

// full run
ann_FF_run(x,N,W)

// encoder
encoder = ann_FF_run(x,N,W,[2,2])
// decoder
decoder = ann_FF_run(encoder,N,W,[3,3])

