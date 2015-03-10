echo Starting
python extractor_word.py -wm d:\dp\data\wordmatrixsenna.npy -wl d:\ss\wordlistsenna.pkl -out d:\ss\kb\sense_kb\kb_senna.pkl
python extractor_word.py -wm d:\dp\data\wordmatrixhlblnew.npy -wl d:\ss\wordlisthlbl.pkl -out d:\ss\kb\sense_kb\kb_hlbl.pkl
python extractor_word.py -wm d:\dp\data\wordmatrix_doc.npy -wl d:\ss\wordlist_doc2dep.pkl -out d:\ss\kb\sense_kb\kb_doc.pkl
python extractor_word.py -wm d:\dp\data\wordmatrix_dep.npy -wl d:\ss\wordlist_doc2dep.pkl -out d:\ss\kb\sense_kb\kb_dep.pkl
python extractor_word.py -wm d:\dp\data\wordmatrix_skip_hs.npy -wl d:\ss\wordlist_skip_hs.pkl -out d:\ss\kb\sense_kb\kb_skip.pkl
python extractor_word.py -wm d:\dp\data\wordmatrix_glove_300.npy -wl d:\ss\wordlist_glove_300.pkl -out d:\ss\kb\sense_kb\kb_glove.pkl
python extractor_word.py -wm d:\dp\data\wordmatrixtomas640.npy -wl d:\ss\wordlisttomas640.pkl -out d:\ss\kb\sense_kb\kb_tomas.pkl
#python combine_dsm_wn_word.py