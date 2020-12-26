dims=(2 4 8 16 32)
for d in "${dims[@]}";
do
  python mvn_conjugate_speed_test.py --ndims=$d --num_live_points=$(echo $d*50 | bc -l)
done
