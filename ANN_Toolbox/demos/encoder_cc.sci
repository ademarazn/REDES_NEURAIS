// Tight 4-2-4 encoder using a mixed standard/conjugate gradients algorithm

rand('seed',0);

x = [1,0,0,0;
     0,1,0,0;
     0,0,1,0;
     0,0,0,1]';

t = x;

N = [4,2,4];

W = ann_FF_init(N, [-1,1], [-1,1]);

// --- standard BP algorithm ---
// learning parameter for standard BP part
lp = [2.5,0];
printf("Standard BP ...");
// standard BP for first 20 steps
T = 20;
W = ann_FF_Std_online(x,t,N,W,lp,T);

// --- Conjugate Gradients algorithm ---
printf("Conjugate Gradients ...");
T = 20;
dW = 0.00001;
W = ann_FF_ConjugGrad(x, t, N, W, T, dW);

// --- test ---

// full run
ann_FF_run(x,N,W)

// encoder
encoder = ann_FF_run(x,N,W,[2,2])
// decoder
decoder = ann_FF_run(encoder,N,W,[3,3])
