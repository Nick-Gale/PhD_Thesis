#bin/bash
tournumber=(10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000);
timesshort=(49 65 89 115 145 180 217 261 302 350 402);
timeslong=(490 650 890 1150 1450 1800 2170 2610 3020 3500 4020);
nl=11;
sam=30;
for i in {0..11};
	do
	for s in {1..30};
		do
		touch /home/nicholas_gale/Documents/Projects/Elastic_Neighbourhood/HeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.EAXGA_short.t
		touch /home/nicholas_gale/Documents/Projects/Elastic_Neighbourhood/HeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.EAXGA_long.t
		echo "">/home/nicholas_gale/Documents/Projects/Elastic_Neighbourhood/HeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.EAXGA_short.t
		echo "">/home/nicholas_gale/Documents/Projects/Elastic_Neighbourhood/HeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.EAXGA_long.t
		eval "./src_short/GA-EAX-restart /home/nicholas_gale/Documents/Projects/Elastic_Neighbourhood/HeuristicComparisons/euc_rand_tour_${tournumber[i]}_$s.tsp 100 30 -1 ${timesshort[i]} 1&"
		eval "./src_long/GA-EAX-restart /home/nicholas_gale/Documents/Projects/Elastic_Neighbourhood/HeuristicComparisons/euc_rand_tour_${tournumber[i]}_$s.tsp 100 30 -1 ${timeslong[i]} 1&"
	done
done



#./src_short/GA-EAX-restart /home/nicholas_gale/Documents/Projects/Elastic_Neighbourhood/HeuristicComparisons/euc_rand_tour_10000_5.tsp 300 30 -1 1 1
