#!/bin/bash

bash ./LKH/tours_generate_parallel.sh
bash ./LKH/tours_cleanup.sh

bash ./GA-EAX/tours_generate_parallel.sh
bash ./GA-EAX/tours_cleanup.sh