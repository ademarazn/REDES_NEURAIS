// ==================================================
// Loose 4-3-4 encoder on a backpropagation network without biases
// (Note that the tight 4-2-4 encoder will not work without biases)
// ==================================================
FILENAMEDEM = "encoder_nb";
scepath = get_absolute_file_path(FILENAMEDEM+".sce");
exec(scepath+FILENAMEDEM+".sci",1);
clear scepath;
clear FILENAMEDEM;
// ==================================================
