CUDA_VISIBLE_DEVICES=0,1 python main.py \
        --config-file configs/vitmatte_s_am2k_100ep.py \
        --num-gpus 2 \
        --num-machines 1 \
        --dist-url "tcp://127.0.0.1:1416"

