// ==================================================
// Loose 8-5-8 encoder
// on a backpropagation network without biases, with SuperSAB
// (Note that the tight 8-3-8 encoder will not work without biases)
// (The 8-4-8 encoder have proven very difficult to train on SuperSAB)
// ==================================================
FILENAMEDEM = "enc848_ssab";
scepath = get_absolute_file_path(FILENAMEDEM+".sce");
exec(scepath+FILENAMEDEM+".sci",1);
clear scepath;
clear FILENAMEDEM;
// ==================================================
