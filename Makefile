.PHONY: train_shannon train_gpu train_mix continue_gpu continue_shannon clean test_bocA test_bocB test_boc_mix

train_gpu:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu -c 4 -s train.py -- --dataroot ./datasets/bolo2imagenet_320 --name bolo2imagenet_320 --model cycle_gan --load_size 320 --crop_size 320 

train_shannon:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu-shannon -c 4 -s train.py -- --dataroot ./datasets/bolo2imagenet_320 --name bolo2imagenet_320 --model cycle_gan --load_size 320 --crop_size 320 

train_mix:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu-shannon -c 4 -s train.py -- --dataroot ./datasets/boc2boc_mix --name boc2boc_mix_320 --model cycle_gan --load_size 320 --crop_size 320 

continue_gpu:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu -c 4 -s train.py -- --dataroot ./datasets/boloA2imagenet_320 --name boloA2imagenet_320 --model cycle_gan --continue_train --epoch_count 101 --load_size 320 --crop_size 320

continue_shannon:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p gpu-shannon -c 4 -s train.py -- --dataroot ./datasets/boloB2imagenet_320 --name boloB2imagenet_320 --model cycle_gan --continue_train --epoch_count 101 --load_size 416 --crop_size 320

clean:
	rm slurm-*

test_boc:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p cpu -c 4 -s test.py -- --dataroot datasets/bolo2imagenet_320/testA --name bolo2imagenet_320 --model test --no_dropout --load_size 320 --crop_size 320 --num_test 409

test_bocB:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p cpu -c 4 -s test.py -- --dataroot datasets/boloB2imagenet_320/testA --name boloB2imagenet_320 --model test --no_dropout --load_size 320 --crop_size 320 --num_test 409

test_boc_mix:
	bash /opt/local/bin/run_py_job.sh -e pytorch-CycleGAN-and-pix2pix -p cpu -c 4 -s test.py -- --dataroot datasets/boc2boc_mix/testA --name boc2boc_mix_320 --model test --no_dropout --load_size 320 --crop_size 320 --num_test 409
