"""
1. Mother script for testing and training the agent for various parameter values
	
	1. Seasons - summer, winter, others (day_chunk + start_date)
	2. Pricing - TOU, dynamic, On-peak/Off-peak, percentage increasing in price (price_scheme = price sensitivity)
	3. Battery Parameters - Size, DOD (P_cap, E_cap) 
	4. Household Peak Load - [3kW, 10kW] - scale base dataset and read same file as input. 

2. With prediction and Without Prediction  - every graph must have two plots (with and without prediction model)	

"""
import main as m
import sys 

def main(type_flag):
	
	eta = 0.9 #efficiency of battery
	DOD = 0.2 #depth of discharge for battery
	total_years = 2 #total training years
	epsilon = 0.8 #exploration control 

	if type_flag == 1:
		#please make canges to the other main like passing pricing_shceme as an argument to the environment class
		summer = (334, 30) #This is April, the hottest month. Full (April, May, June, July) [334, 94]  
		winter = (261, 30) #This is Feb, the coldest month. Full (Dec, Jan, Feb) [212,303]
		#others = (95,30) # (Aug, Sep, Oct, Nov and March) [95, 211] [303,333]
		E_cap = 6.4 
		P_cap = 3.0
		#pricing_scheme = [.040,.040,.040,.040,.040,.040,.080,.080,.080,.080,.040,.040,.080,.080,.080,.040,.040,.120,.120,.040,.040,.040,.040,.040]
		summer_price = [.087,.087,.087,.087,.087,.087,0.087,.132,.132,.132,.132,.18,.18,.18,.18,.18,.18,.132,.132, 0.087,.087,.087,.087,0.087]
		winter_price = [.087,.087,.087,.087, 0.087, 0.087, 0.087, 0.18, 0.18, 0.18, 0.18,.132,.132,.132,.132,.132,.132,.18,.18,.087,.087,.087,.087,0.087]
		name = '_summer' #identifier for saving the model
		m.main(summer[0], summer[1], eta, E_cap, P_cap, epsilon, total_years, summer_price, DOD, name)
		name = '_winter'
		m.main(winter[0], winter[1], eta, E_cap, P_cap, epsilon, total_years, winter_price, DOD, name)
	
	elif type_flag == 3:
		"""
		1. Dynamic Pricing - LATER
		2. Savings Vs peak-to-off-peak price ratio (while keeping the average price constant) for a X kWh capacity battery.
		"""
		#dyna_price = [0.007, 0.016, 0.011, 0.021, 0.007, 0.011, 0.01, 0.019, 0.01, 0.006, 0.02, 0.013, 0.021, 0.007, 0.017, 0.02, 0.005, 0.008, 0.021, 0.011, 0.017, 0.024, 0.01, 0.01]
		#price = [.0040,.0040,.0040,.0040,.0040,.0040,.0080,.0080,.0080,.0080,.0040,.0040,.0080,.0080,.0080,.0040,.0040,.0120,.0120,.0040,.0040,.0040,.0040,.0040]
		price = [.087,.087,.087,.087,.087,.087,0.087,.132,.132,.132,.132,.18,.18,.18,.18,.18,.18,.132,.132, 0.087,.087,.087,.087,0.087]
		
		#dyna_price generated using np.uniform(low=0.087, high=0.24)
		dyna_price = [0.09, 0.089, 0.147, 0.16, 0.191, 0.14, 0.168, 0.193, 0.089, 0.221, 0.225, 0.198, 0.147, 0.131, 0.207, 0.088, 0.214, 0.184, 0.1, 0.163, 0.157, 0.188, 0.147, 0.126]
		eta = 0.9
		E_cap = 6.4 
		P_cap = 3.0
		total_years = 2
		limits = (334,30)
		name = '_dynamic_pricing'
		E_cap = 6.4 
		P_cap = 3.0
		#for i, ratio in enumerate(ratio_list):
		#	eq_ratio = (1.4 - 0.024*ratio)/0.56 #ratio for adjusting time-slots to keep average price constant.
		#	name = name + str(i)
		#	m.main(limits[0], limits[1], eta, E_cap, P_cap, epsilon, total_years, pricing_scheme, DOD, name)
		m.main(limits[0], limits[1], eta, E_cap, P_cap, epsilon, total_years, dyna_price, DOD, name)
		
	elif type_flag == 2:
		"""
		Vary E_cap = [6,9,12,18,24,30] and keep the charging rate (P_cap) constant.
		"""
		pricing_scheme = [.040,.040,.040,.040,.040,.040,.080,.080,.080,.080,.040,.040,.080,.080,.080,.040,.040,.120,.120,.040,.040,.040,.040,.040]
		energy_rating = [6,9,12,15,18,21,24,30]
		P_cap = 3.0
		limits = (60,30)
		name = '_e_cap'
		for E_cap in energy_rating:
			name = name + str(E_cap)
			m.main(limits[0], limits[1], eta, E_cap, P_cap, epsilon, total_years, pricing_scheme, DOD, name)
		
if __name__ == '__main__' :
	#for type_flag in [1, 2, 3] :
	main(int(sys.argv[1]))



