#! /usr/bin/zsh
for i in {-5..0}
do
    python gen_noise_low_light.py --EV $i
done