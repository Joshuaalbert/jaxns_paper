dims=(1 2 3 4 5 6)
K=10
for d in "${dims[@]}";
do
  python gaussian_mixture_speed_test.py --K=$K --ndims=$d --num_live_points=$(echo $d*50*$K | bc -l)
done
