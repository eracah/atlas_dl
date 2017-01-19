
import matplotlib; matplotlib.use("agg")


d = {'/global/cscratch1/sd/racah/atlas_h5/jetjet_JZ10.h5': 261937,
 '/global/cscratch1/sd/racah/atlas_h5/jetjet_JZ11.h5': 92308,
 '/global/cscratch1/sd/racah/atlas_h5/jetjet_JZ3.h5': 27,
 '/global/cscratch1/sd/racah/atlas_h5/jetjet_JZ4.h5': 84112,
 '/global/cscratch1/sd/racah/atlas_h5/jetjet_JZ5.h5': 459224,
 '/global/cscratch1/sd/racah/atlas_h5/jetjet_JZ6.h5': 493364,
 '/global/cscratch1/sd/racah/atlas_h5/jetjet_JZ7.h5': 509473,
 '/global/cscratch1/sd/racah/atlas_h5/jetjet_JZ8.h5': 445892,
 '/global/cscratch1/sd/racah/atlas_h5/jetjet_JZ9.h5': 207695,
 '/global/cscratch1/sd/racah/atlas_h5/test_jetjet_JZ10.h5': 261934,
 '/global/cscratch1/sd/racah/atlas_h5/test_jetjet_JZ11.h5': 92303,
 '/global/cscratch1/sd/racah/atlas_h5/test_jetjet_JZ3.h5': 0,
 '/global/cscratch1/sd/racah/atlas_h5/test_jetjet_JZ4.h5': 84112,
 '/global/cscratch1/sd/racah/atlas_h5/test_jetjet_JZ5.h5': 459218,
 '/global/cscratch1/sd/racah/atlas_h5/test_jetjet_JZ6.h5': 493365,
 '/global/cscratch1/sd/racah/atlas_h5/test_jetjet_JZ7.h5': 509474,
 '/global/cscratch1/sd/racah/atlas_h5/test_jetjet_JZ8.h5': 445884,
 '/global/cscratch1/sd/racah/atlas_h5/test_jetjet_JZ9.h5': 207691}



import pickle



pickle.(d, file=open("/global/cscratch1/sd/racah/atlas_h5/file_max_inds.pkl", "w"))








