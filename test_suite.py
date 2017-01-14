"""

1. Mother script for testing and training the agent for various parameter values
	
	1. Seasons - summer, winter, others (day_chunk + start_date)
	2. Pricing - TOU, dynamic, On-peak/Off-peak, percentage increasing in price (price_scheme = price sensitivity)
	3. Battery Parameters - Size, DOD (P_cap, E_cap) 
	4. Household Peak Load - [3kW, 10kW] - scale base dataset and read same file as input. 

2. With prediction and Without Prediction  - every graph must have two plots (with and without prediction model)	

"""
import main as m

def main(type_flag):
	
	if type_flag == 1:
		#please make canges to the other main like passing pricing_shceme as an argument to the environment class
		summer = (0, 30)
		winter = (275, 30)
		others = () # please fill with proper start and duration
		eta = 0.9
		E_cap = 6.4 
		P_cap = 3.0
		total_years = 5
		price = [.040,.040,.080,.080,.120,.240,.120,.040,.040,.040,.040,.080,.120,.080,.120,.040,.040,.120,.120,.040,.040,.040,.040,.040]
		DOD = 0.1
		m.main(summer[0], summer[1], eta, E_cap, P_cap, epsilon, total_years, pricing_scheme, DOD)
		m.main(winter[0], winter[1], eta, E_cap, P_cap, epsilon, total_years, pricing_scheme, DOD)
	elif type_flag == 2:

	elif type_flag == 3:

	else:

		
if __name__ == '__main__' :
	for type in [1, 2, 3, 4] :
		main(type)


