.PHONY: shannon cpu clean:

shannon:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu-shannon -c 4 -s train.py -- --dataroot ./datasets/corals2nice_corals --name c2c_run2 --model cycle_gan --n_epochs 5 --n_epochs_decay 5

cpu:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p cpu -c 4 -s train.py -- --dataroot ./datasets/corals2nice_corals --name c2c_run2 --model cycle_gan --n_epochs 5 --n_epochs_decay 5

clean:
	rm slurm-*