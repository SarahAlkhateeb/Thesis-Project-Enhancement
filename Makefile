.PHONY: train_cpu train_gpu continue clean test

train_gpu:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu -c 8 -s train.py -- --dataroot ./datasets/coral2coral --name c2c_cyclegan_256 --model cycle_gan 

train_shannon:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu-shannon -c 8 -s train.py -- --dataroot ./datasets/coral2coral --name c2c_cyclegan_416 --model cycle_gan --load_size 416 --crop_size 416

continue:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p cpu -c 4 -s train.py -- --dataroot ./datasets/coral2coral --name c2c_cyclegan --model cycle_gan --continue_train --epoch_count 101

clean:
	rm slurm-*

test_256:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p cpu -c 4 -s test.py -- --dataroot datasets/coral2coral/testA --name c2c_cyclegan_256 --model test --no_dropout

test_416:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p cpu -c 4 -s test.py -- --dataroot datasets/coral2coral/testA --name c2c_cyclegan_416 --model test --no_dropout --load_size 416 --crop_size 416
