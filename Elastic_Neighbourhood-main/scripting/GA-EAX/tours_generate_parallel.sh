#bin/bash
tournumber=(10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000);
timesshort=(49 65 89 115 145 180 217 261 302 350 402);
timesshort=(25 32 40 57 72 90 110 130 150 175 201);
timeslong=(490 650 890 1150 1450 1800 2170 2610 3020 3500 4020);
nl=11;
sam=30;
for i in {0..11};
	do
	for s in {1..10};
		do
		touch ./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.EAXGA_short.t
		echo "">./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.EAXGA_short.t
		eval "./src_short/GA-EAX-restart ./../../dataHeuristicComparisons/euc_rand_tour_${tournumber[i]}_$s.tsp 100 30 -1 ${timesshort[i]} 1&"
	done
	wait
	for s in {11..20};
		do
		touch ./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.EAXGA_short.t
		echo "">./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.EAXGA_short.t
		eval "./src_short/GA-EAX-restart ./../../dataHeuristicComparisons/euc_rand_tour_${tournumber[i]}_$s.tsp 100 30 -1 ${timesshort[i]} 1&"
	done
	wait
	
	for s in {21..30};
		do
		touch ./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.EAXGA_short.t
		echo "">./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.EAXGA_short.t
		eval "./src_short/GA-EAX-restart ./../../dataHeuristicComparisons/euc_rand_tour_${tournumber[i]}_$s.tsp 100 30 -1 ${timesshort[i]} 1&"
	done
	wait
done


for i in {0..11};
	do
	for s in {1..10};
		do
		touch ./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.EAXGA_long.t
		echo "">./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.EAXGA_long.t
		eval "./src_long/GA-EAX-restart ./../../dataHeuristicComparisons/euc_rand_tour_${tournumber[i]}_$s.tsp 100 30 -1 ${timeslong[i]} 1&"
	done
	wait
	for s in {11..20};
		do
		touch ./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.EAXGA_long.t
		echo "">./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.EAXGA_long.t
		eval "./src_long/GA-EAX-restart ./../../dataHeuristicComparisons/euc_rand_tour_${tournumber[i]}_$s.tsp 100 30 -1 ${timeslong[i]} 1&"
	done
	wait
	
	for s in {21..30};
		do
		touch ./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.EAXGA_long.t
		echo "">./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.EAXGA_long.t
		eval "./src_long/GA-EAX-restart ./../../dataHeuristicComparisons/euc_rand_tour_${tournumber[i]}_$s.tsp 100 30 -1 ${timeslong[i]} 1&"
	done
	wait
done


