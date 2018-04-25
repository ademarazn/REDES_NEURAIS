// ==================================================
// Loose 8-4-8 encoder
// on a backpropagation network without biases, with momentum
// (Note that the tight 8-4-8 encoder will not work without biases)
// ==================================================
FILENAMEDEM = "enc848_m_nb";
scepath = get_absolute_file_path(FILENAMEDEM+".sce");
exec(scepath+FILENAMEDEM+".sci",1);
clear scepath;
clear FILENAMEDEM;
// ==================================================
