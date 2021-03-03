.PHONY: train_cpu train_gpu continue_gpu continue_shannon clean test_boc test_boc_mix

train_gpu:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu -c 4 -s train.py -- --dataroot ./datasets/boc2boc --name b2b_cyclegan_416 --model cycle_gan --load_size 416 --crop_size 416

train_shannon:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu-shannon -c 4 -s train.py -- --dataroot ./datasets/boc2boc_mix --name b2b_mix_cyclegan_416 --model cycle_gan --load_size 416 --crop_size 416

continue_gpu:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu -c 4 -s train.py -- --dataroot ./datasets/boc2boc --name b2b_cyclegan_416 --model cycle_gan --continue_train --epoch_count 101 --load_size 416 --crop_size 416

continue_shannon:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu-shannon -c 4 -s train.py -- --dataroot ./datasets/boc2boc_mix --name b2b_mix_cyclegan_416 --model cycle_gan --continue_train --epoch_count 101 --load_size 416 --crop_size 416

clean:
	rm slurm-*

test_boc:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p cpu -c 4 -s test.py -- --dataroot datasets/boc2boc/testA --name b2b_cyclegan_416 --model test --no_dropout --load_size 416 --crop_size 416 --num_test 409

test_boc_mix:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p cpu -c 4 -s test.py -- --dataroot datasets/boc2boc_mix/testA --name b2b_mix_cyclegan_416 --model test --no_dropout --load_size 416 --crop_size 416 --num_test 409
