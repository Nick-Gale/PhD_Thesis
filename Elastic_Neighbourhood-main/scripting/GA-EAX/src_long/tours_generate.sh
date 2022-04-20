#do the short tours

tournumber=(10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000);
timesshort=(40 40 40 40 40 40 40 40 40 40 40);
nl=11;
sam=5;
for i in {0..$nl};
	do
	for s in {1..$sam};
		do
		echo parameters_short_tn${tournumber[i]}_s$s.par.tsp
		echo PROBLEM_FILE = /home/nicholas_gale/Documents/Projects/Elastic_Neighbourhood/HeuristicComparisons/euc_rand_tour_${tournumber[i]}_$s.tsp >> parameters_short_tn${tournumber[i]}_s$s.par.tsp
		echo RUNS = 1 >> parameters_short_tn${tournumber[i]}_s$s.par.tsp
		echo TIME_LIMIT = ${timesshort[i]} >> parameters_short_tn${tournumber[i]}_s$s.par.tsp
		echo INITIAL_PERIOD = 200 >> parameters_short_tn${tournumber[i]}_s$s.par.tsp
		echo TOUR_FILE = euc_rand_tour_${tournumber[i]}_$s.LKH_short.tour >> parameters_short_tn${tournumber[i]}_s$s.par.tsp
		./LKH parameters_short_tn${tournumber[i]}_s$s.par.tsp &
	done
done

