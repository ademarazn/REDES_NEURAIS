// ==================================================
// Tight 4-2-4 encoder
// on a backpropagation ANN with biases and momentum
// ==================================================
FILENAMEDEM = "encoder_m";
scepath = get_absolute_file_path(FILENAMEDEM+".sce");
exec(scepath+FILENAMEDEM+".sci",1);
clear scepath;
clear FILENAMEDEM;
// ==================================================
