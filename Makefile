.PHONY: train_cpu train_gpu continue_gpu continue_shannon clean test_bocA test_bocB

train_gpu:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu -c 4 -s train.py -- --dataroot ./datasets/boloA2imagenet_320 --name boloA2imagenet_320 --model cycle_gan --load_size 320 --crop_size 320 --n_epochs 50 n_epochs_decay 50

train_shannon:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu-shannon -c 4 -s train.py -- --dataroot ./datasets/boloB2imagenet_320 --name boloB2imagenet_320 --model cycle_gan --load_size 320 --crop_size 320 --n_epochs 50 n_epochs_decay 50

continue_gpu:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu -c 4 -s train.py -- --dataroot ./datasets/boloA2imagenet_320 --name boloA2imagenet_320 --model cycle_gan --continue_train --epoch_count 101 --load_size 320 --crop_size 320

continue_shannon:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu-shannon -c 4 -s train.py -- --dataroot ./datasets/boloB2imagenet_320 --name boloB2imagenet_320 --model cycle_gan --continue_train --epoch_count 101 --load_size 416 --crop_size 320

clean:
	rm slurm-*

test_bocA:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p cpu -c 4 -s test.py -- --dataroot datasets/boloA2imagenet_320/testA --name boloA2imagenet_320 --model test --no_dropout --load_size 320 --crop_size 320 --num_test 409

test_bocB:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p cpu -c 4 -s test.py -- --dataroot datasets/boloB2imagenet_320/testA --name boloB2imagenet_320 --model test --no_dropout --load_size 320 --crop_size 320 --num_test 409
