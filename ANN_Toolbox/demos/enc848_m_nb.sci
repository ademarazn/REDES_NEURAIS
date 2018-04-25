// Loose 8-4-8 encoder
// on a backpropagation network without biases, with momentum
// (Note that the tight 8-4-8 encoder will not work without biases)

rand('seed',0);

// network def.
//  - neurons per layer, including input
N  = [8,4,8];

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
lp = [2.5,0.1,0.9,0.25];

// init randomize weights between:
r = [-1,7];

W = ann_FF_init_nb(N,r);
Delta_W_old = hypermat(size(W)');

// 250 epochs are enough to ilustrate
T = 250;
[W,Delta_W_old] = ann_FF_Mom_online_nb(x,t,N,W,lp,T,Delta_W_old);

// full run
ann_FF_run_nb(x,N,W)

// encoder
encoder = ann_FF_run_nb(x,N,W,[2,2])
// decoder
decoder = ann_FF_run_nb(encoder,N,W,[3,3])

