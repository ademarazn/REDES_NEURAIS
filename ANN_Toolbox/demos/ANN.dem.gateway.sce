// ====================================================================
// Copyright INRIA 2008
// Copyright DIGITEO 2011
// Allan CORNET
// ====================================================================
function subdemolist = demo_ANN_gw()

demopath = get_absolute_file_path("ANN.dem.gateway.sce");

subdemolist = [ "encoder 4-3-4 on ANN without biases",                                          "encoder_nb.sce"      ; ..
     "tight encoder 4-2-4 on ANN with biases",                                                  "encoder.sce"         ; ..
     "encoder 4-3-4 on ANN without biases compare with encoder_nb.sce",                         "encoder_m_nb.sce"    ; ..
     "tight encoder 4-2-4 on ANN with biases compare with encoder.sce",                         "encoder_m.sce"       ; ..
     "encoder 8-4-8 on ANN without biases",                                                     "enc848_m_nb.sce"     ; ..
     "encoder 8-3-8 on ANN with biases",                                                        "enc838_m.sce"        ; ..
     "encoder 8-5-8 on ANN without biases",                                                     "enc858_ssab_nb.sce"  ; ..
     "encoder 8-4-8 on ANN with biases",                                                        "enc848_ssab.sce"     ; ..
     "tight encoder 4-2-4 on ANN with biases uses a mixed standard/conjugate gradients method", "encoder_cc.sce" ..
     ];

subdemolist(:,2) = demopath + subdemolist(:,2);
endfunction
// ====================================================================
subdemolist = demo_ANN_gw();
clear demo_ANN_gw;
// ====================================================================
