// Loose 4-3-4 encoder on a backpropagation network without biases
// (Note that the tight 4-2-4 encoder will not work without biases)

// ensure the same random starting point
rand('seed',0);

// network def.
//  - neurons per layer, including input
N  = [4,3,4];

// inputs
x = [1,0,0,0;
     0,1,0,0;
     0,0,1,0;
     0,0,0,1]';

// targets, at training stage is acts as identity network
t = x;

// learning parameter
lp = [8,0];

// init randomize weights between
r = [-1,1];

W = ann_FF_init_nb(N,r);

// 500 epochs are enough to ilustrate
T = 500;
W = ann_FF_Std_online_nb(x,t,N,W,lp,T);

// full run
ann_FF_run_nb(x,N,W)

// encoder
encoder = ann_FF_run_nb(x,N,W,[2,2])
// decoder
decoder = ann_FF_run_nb(encoder,N,W,[3,3])

