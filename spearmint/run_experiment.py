import prepare_experiment as pe

#some fixed parameters for that run
params_fixed=dict(
    #distributed training
    distributed_training=True,
    #number of threads:
    num_threads=32,
    #network name
    network_name="hep-improved-classifier",
    #batch size ~ sqrt(1094)
    batch_size_train=32,
    #maximal 50 epochs
    max_iter=1700,
    #max_iter=170,
    iters_per_epoch=34,
    #for validation, batchsize 8 is good
    batch_size_validation=8,
    num_iters_validation=8,
    #network parameters
    num_conv=5,
    num_filters=64,
    num_nodes=1024,
    #early stopping parameters:
    #do 5 epochs regardless
    patience_iter = 170,
    #increase by 3 epochs if good
    patience_increase_iter = 102,
    #decide by that threshold
    improvement_threshold = 0.995
)

# Write a function like this called 'main'
def main(job_id, params):
    print("Starting experiment ",job_id)

    #create directory for job
    mainpath="/global/cscratch1/sd/tkurth/atlas_dl/atlas_caffe/spearmint_optimization"
    expath=mainpath+"/experiments/job_"+str(job_id)

    #compute index for the files:
    fileindex=job_id%params_fixed["num_nodes"]

    #spearmint parameters
    params_spearmint=dict(
                fileindex=fileindex,
                experiment_path=expath,
                filename_runscript="runscript.sh",
                filename_solver="solver.prototxt",
                filename_train_net="train.prototxt",
                filename_validation_net="validation.prototxt",
                filename_snapshot_prefix="hep_improved_model",
                filename_train_data="train.txt",
                filename_validation_data="validation.txt",
                data_path=mainpath+'/data',
                caffe_path='/project/projectdirs/mpccc/tkurth/NESAP/intelcaffe_internal/install_cori-hsw-nompi'
                )
    print("Current parameters: ",params)

    #prepare infrastructure
    print("Creating job folder.")
    pe.create_folder_structure(expath)

    #prepare input files
    print("Creating input files.")
    pe.create_input_files(params_spearmint)

    #prepare runscript
    print("Creating runscript.")
    pe.create_runscript(params_spearmint)

    #prepare solver file
    print("Preparing solver.")
    solver=pe.create_solver_file(params_spearmint,params,params_fixed)

    #prepare network file
    print("Preparing network.")
    networks=pe.create_network_file(params_spearmint,params,params_fixed)

    #run training
    print("Run training.")
    if params_fixed["distributed_training"]:
        return pe.run_distributed_training(params_spearmint,params_fixed)
    else:
        return pe.run_training(params_spearmint,params_fixed)
