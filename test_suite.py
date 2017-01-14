"""

1. Mother script for testing and training the agent for various parameter values
	
	1. Seasons - summer, winter, others (day_chunk + start_date)
	2. Pricing - TOU, dynamic, On-peak/Off-peak, percentage increasing in price (price_scheme = price sensitivity)
	3. Battery Parameters - Size, DOD (P_cap, E_cap) 
	4. Household Peak Load - [3kW, 10kW] - scale base dataset and read same file as input. 

2. With prediction and Without Prediction  - every graph must have two plots (with and without prediction model)	

"""
import main

def main(type_flag):
	if type_flag == 1:
		summer = (0, 30)
		winter = (275, 30)
		others = ()
	elif type_flag == 2:

	elif type_flag == 3:

	else:


