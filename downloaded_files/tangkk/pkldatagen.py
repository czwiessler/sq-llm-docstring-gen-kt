from acesongdb import pkl_data_nseg_h5py, pkl_data_framewise_h5py, pkl_data_nseg, pkl_data_framewise, pkl_data_varlen, pkl_data_varlen_h5py, pkl_data_matrix, pkl_data_waveform_h5py

####################################### ch ############################################

# ymax = 277 #inv
# pkl_data_framewise('../data/ch/Jsong-ch-inv.mat','../data/ch/Jsong-ch-inv.pkl', ymax)
# pkl_data_framewise('../data/ch/CJsong-ch-inv.mat','../data/ch/CJsong-ch-inv.pkl', ymax)
# pkl_data_framewise('../data/ch/CJKsong-ch-inv.mat','../data/ch/CJKsong-ch-inv.pkl', ymax)
# pkl_data_framewise('../data/ch/Bsong-ch-inv.mat','../data/ch/Bsong-ch-inv.pkl', ymax)
# pkl_data_framewise_h5py('../data/ch/Usong-ch-inv.mat','../data/ch/Usong-ch-inv.pkl', ymax)
# pkl_data_framewise_h5py('../data/ch/CJKUsong-ch-inv.mat','../data/ch/CJKUsong-ch-inv.pkl', ymax)
# pkl_data_framewise_h5py('../data/ch/CJKURsong-ch-inv.mat','../data/ch/CJKURsong-ch-inv.pkl', ymax) #U is U191
# pkl_data_nseg('../data/ch/Jsong-ch-inv.mat','../data/ch/Jsong-6seg-ch-inv.pkl',6, ymax)
# pkl_data_nseg('../data/ch/Bsong-ch-inv.mat','../data/ch/Bsong-6seg-ch-inv.pkl',6, ymax)

# ymax = 61 #noinv
# pkl_data_framewise('../data/ch/Jsong-ch-noinv.mat','../data/ch/Jsong-ch-noinv.pkl',ymax)
# pkl_data_framewise('../data/ch/Bsong-ch-noinv.mat','../data/ch/Bsong-ch-noinv.pkl',ymax)
# pkl_data_nseg('../data/ch/Jsong-ch-noinv.mat','../data/ch/Jsong-6seg-ch-noinv.pkl',6,ymax)
# pkl_data_nseg('../data/ch/Bsong-ch-noinv.mat','../data/ch/Bsong-6seg-ch-noinv.pkl',6,ymax)

# ymax = 73 #no7
# pkl_data_framewise('../data/ch/Jsong-ch-no7.mat','../data/ch/Jsong-ch-no7.pkl', ymax)
# pkl_data_framewise('../data/ch/Bsong-ch-no7.mat','../data/ch/Bsong-ch-no7.pkl', ymax)
# pkl_data_nseg('../data/ch/Jsong-ch-no7.mat','../data/ch/Jsong-6seg-ch-no7.pkl',6, ymax)
# pkl_data_nseg('../data/ch/Bsong-ch-no7.mat','../data/ch/Bsong-6seg-ch-no7.pkl',6, ymax)

# no need to pkl pure matrix .mat file for the moment, since it can be handled quite nicely in scipy.io
# pkl_data_matrix('../data/ch/J6seg-ch-inv.mat','../J6seg-ch-inv.pkl')
# pkl_data_matrix('../data/ch/J6seg-ch-noinv.mat','../J6seg-ch-noinv.pkl')
# pkl_data_matrix('../data/ch/B6seg-ch-inv.mat','../B6seg-ch-inv.pkl')
# pkl_data_matrix('../data/ch/B6seg-ch-noinv.mat','../B6seg-ch-noinv.pkl')

# pkl variable length data
# pkl_data_varlen('../data/ch/Jvarlen-ch-inv.mat','../data/ch/J6varlen-ch-inv.pkl',24,1)
# pkl_data_varlen('../data/ch/Jvarlen-ch-noinv.mat','../data/ch/J6varlen-ch-noinv.pkl',24,1)
# pkl_data_varlen('../data/ch/Bvarlen-ch-inv.mat','../data/ch/B6varlen-ch-inv.pkl',24,1)
# pkl_data_varlen('../data/ch/Bvarlen-ch-noinv.mat','../data/ch/B6varlen-ch-noinv.pkl',24,1)
# pkl_data_varlen('../data/ch/CJKvarlen-ch-inv.mat','../data/ch/CJKvarlen-ch-inv.pkl',24,1)

####################################### ns ############################################

# ymax = 277 #inv
# pkl_data_framewise_h5py('../data/ns/Jsong-ns-inv.mat','../data/ns/Jsong-ns-inv.pkl', ymax)
# pkl_data_nseg_h5py('../data/ns/Jsong-ns-inv.mat','../data/ns/Jsong-6seg-ns-inv.pkl',6, ymax)
# pkl_data_framewise_h5py('../data/ns/CJsong-ns-inv.mat','../data/ns/CJsong-ns-inv.pkl', ymax)
# pkl_data_framewise_h5py('../data/ns/CJKsong-ns-inv.mat','../data/ns/CJKsong-ns-inv.pkl', ymax)
# pkl_data_framewise_h5py('../data/ns/Usong-ns-inv.mat','../data/ns/Usong-ns-inv.pkl', ymax)
# pkl_data_framewise_h5py('../data/ns/CJKUsong-ns-inv.mat','../data/ns/CJKUsong-ns-inv.pkl', ymax)

# pkl_data_varlen_h5py('../data/ns/CJKvarlen-ns-inv.mat','../data/ch/CJKvarlen-ns-inv.pkl',252,1)

# ymax = 61 #noinv
# pkl_data_framewise_h5py('../data/ns/Jsong-ns-noinv.mat','../data/ns/Jsong-ns-noinv.pkl',ymax)
# pkl_data_nseg_h5py('../data/ns/Jsong-ns-noinv.mat','../data/ns/Jsong-6seg-ns-noinv.pkl',6,ymax)

# ymax = 73 #no7
# pkl_data_framewise_h5py('../data/ns/Jsong-ns-no7.mat','../data/ns/Jsong-ns-no7.pkl', ymax)
# pkl_data_nseg_h5py('../data/ns/Jsong-ns-no7.mat','../data/ns/Jsong-6seg-ns-no7.pkl',6, ymax)

####################################### sg ############################################
# ymax = 277 #inv
# pkl_data_framewise_h5py('../data/sg/Jsong-sg-inv.mat','../data/sg/Jsong-sg-inv.pkl', ymax)

####################################### wf ############################################
# ymax = 277 #inv
# pkl_data_waveform_h5py('../data/wf/Jsong-wf-inv.mat','../data/wf/Jsong-wf-inv.pkl', ymax)

####################################### jazz ############################################
# ymax = 421 # jazz
# pkl_data_framewise('../data/ch/Jazzsong-ch-inv.mat','../data/ch/Jazzsong-ch-inv.pkl', ymax)
# pkl_data_framewise('../data/ns/Jazzsong-ns-inv.mat','../data/ns/Jazzsong-ns-inv.pkl', ymax)

#ymax = 481 # jazz all
#pkl_data_framewise('../data/ch/JazzTutorialsong-ch-inv.mat','../data/ch/JazzTutorialsong-ch-inv.pkl', ymax)
#pkl_data_waveform_h5py('../data/ns/JazzTutorialsong-ns-inv.mat','../data/ns/JazzTutorialsong-ns-inv.pkl', ymax)