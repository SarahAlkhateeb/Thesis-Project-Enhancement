.PHONY: train continue clean test

train:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p cpu -c 4 -s train.py -- --dataroot ./datasets/coral2coral --name c2c_cyclegan --model cycle_gan --n_epochs 50 --n_epochs_decay 50

continue:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p cpu -c 4 -s train.py -- --dataroot ./datasets/coral2coral --name c2c_cyclegan --model cycle_gan --continue_train --epoch_count 101

clean:
	rm slurm-*

test:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p cpu -c 4 -s test.py -- --dataroot datasets/corals2coral/testA --name c2c_cyclegan --model test --no_dropout
