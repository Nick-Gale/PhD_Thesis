#!/bin/bash
tournumber=(10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000);
timesshort=(25 32 40 57 72 90 110 130 150 175 201);
timeslong=(490 650 890 1150 1450 1800 2170 2610 3020 3500 4020);
nl=11;
sam=5;
for i in {0..10};
	do
	for s in {1..15};
		do
		echo parameters_short_tn${tournumber[i]}_s$s.par.tsp
		echo PROBLEM_FILE = ./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp >> parameters_short_tn${tournumber[i]}_s$s.par.tsp
		echo RUNS = 1 >> parameters_short_tn${tournumber[i]}_s$s.par.tsp
		echo TIME_LIMIT = ${timesshort[i]} >> parameters_short_tn${tournumber[i]}_s$s.par.tsp
		echo INITIAL_PERIOD = 200 >> parameters_short_tn${tournumber[i]}_s$s.par.tsp
		echo TOUR_FILE = ./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.LKH_short.t >> parameters_short_tn${tournumber[i]}_s$s.par.tsp
		echo "" > ./../../Elastic_Neighbourhood/dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.LKH_short.t
		./LKH parameters_short_tn${tournumber[i]}_s$s.par.tsp &
		
		
		echo parameters_long_tn${tournumber[i]}_s$s.par.tsp
		echo PROBLEM_FILE = ./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp >> parameters_long_tn${tournumber[i]}_s$s.par.tsp
		echo RUNS = 1 >> parameters_long_tn${tournumber[i]}_s$s.par.tsp
		echo TIME_LIMIT = ${timeslong[i]} >> parameters_long_tn${tournumber[i]}_s$s.par.tsp
		echo INITIAL_PERIOD = 200 >> parameters_long_tn${tournumber[i]}_s$s.par.tsp
		echo TOUR_FILE = ./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.LKH_long.t >> parameters_long_tn${tournumber[i]}_s$s.par.tsp
		./LKH parameters_long_tn${tournumber[i]}_s$s.par.tsp &
	done
	
	wait 
	
	for s in {16..30};
		do
		echo parameters_short_tn${tournumber[i]}_s$s.par.tsp
		echo PROBLEM_FILE = ./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp >> parameters_short_tn${tournumber[i]}_s$s.par.tsp
		echo RUNS = 1 >> parameters_short_tn${tournumber[i]}_s$s.par.tsp
		echo TIME_LIMIT = ${timesshort[i]} >> parameters_short_tn${tournumber[i]}_s$s.par.tsp
		echo INITIAL_PERIOD = 200 >> parameters_short_tn${tournumber[i]}_s$s.par.tsp
		echo TOUR_FILE = ./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.LKH_short.t >> parameters_short_tn${tournumber[i]}_s$s.par.tsp
		echo "" > ./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.LKH_short.t
		./LKH parameters_short_tn${tournumber[i]}_s$s.par.tsp &
		
		
		echo parameters_long_tn${tournumber[i]}_s$s.par.tsp
		echo PROBLEM_FILE = ./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp >> parameters_long_tn${tournumber[i]}_s$s.par.tsp
		echo RUNS = 1 >> parameters_long_tn${tournumber[i]}_s$s.par.tsp
		echo TIME_LIMIT = ${timeslong[i]} >> parameters_long_tn${tournumber[i]}_s$s.par.tsp
		echo INITIAL_PERIOD = 200 >> parameters_long_tn${tournumber[i]}_s$s.par.tsp
		echo TOUR_FILE = ./../../dataHeuristicComparisons/euc_rand_tour_"${tournumber[i]}"_$s.tsp.LKH_long.t >> parameters_long_tn${tournumber[i]}_s$s.par.tsp
		./LKH parameters_long_tn${tournumber[i]}_s$s.par.tsp &
	done
	
	wait
done

