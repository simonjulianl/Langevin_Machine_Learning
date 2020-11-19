for j in 0.01 0.35 0.85
#for j in 0.01 
do

	for i in $(seq 0 4)
	do
		python animate.py --method vv --sample $i --Temp $j --ts 0.01
	done
done

