// Tight 4-2-4 encoder on a backpropagation ANN

// ensure the same starting point each time
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
lp = [2.5,0];

W = ann_FF_init(N);

// 400 epochs are enough to ilustrate
T = 400;
W = ann_FF_Std_online(x,t,N,W,lp,T);

// full run
ann_FF_run(x,N,W)

// encoder
encoder = ann_FF_run(x,N,W,[2,2])
// decoder
decoder = ann_FF_run(encoder,N,W,[3,3])

