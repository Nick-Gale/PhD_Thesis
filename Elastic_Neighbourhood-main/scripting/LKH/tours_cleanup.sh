#do the short tours

tournumber=(10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000);
timesshort=(40 40 40 40 40 40 40 40 40 40 40);
timesshort=(49 65 89 115 145 180 217 261 302 350 402);
timesshort=(25 32 40 57 72 90 110 130 150 175 201);
timeslong=(490 650 890 1150 1450 1800 2170 2610 3020 3500 4020);
nl=11;
sam=30;
# delete the temporary files
for i in {1..11};
	do
	for s in {0..30};
		do
		rm parameters_short_tn${tournumber[i]}_s$s.par.tsp
		rm parameters_long_tn${tournumber[i]}_s$s.par.tsp
	done
done
