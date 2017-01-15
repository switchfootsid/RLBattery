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
	
	if type_flag == 1:
		#please make canges to the other main like passing pricing_shceme as an argument to the environment class
		summer = (334, 30) #This is April, the hottest month. Full (April, May, June, July) [334, 94]  
		winter = (261, 30) #This is Feb, the coldest month. Full (Dec, Jan, Feb) [212,303]
		#others = (95,30) # (Aug, Sep, Oct, Nov and March) [95, 211] [303,333]
		eta = 0.9
		E_cap = 6.4 
		P_cap = 3.0
		total_years = 5
		price = [.040,.040,.040,.040,.040,.040,.080,.080,.080,.080,.040,.040,.080,.080,.080,.040,.040,.120,.120,.040,.040,.040,.040,.040]
		DOD = 0.2
		m.main(summer[0], summer[1], eta, E_cap, P_cap, epsilon, total_years, pricing_scheme, DOD)
		m.main(winter[0], winter[1], eta, E_cap, P_cap, epsilon, total_years, pricing_scheme, DOD)
	elif type_flag == 2:
		"""
		1. Dynamic Pricing - LATER
		2. Savings Vs peak-to-off-peak price ratio (while keeping the average price constant) for a X kWh capacity battery.
		"""
		#price = [.040,.040,.040,.040,.040,.040,.080,.080,.080,.080,.040,.040,.080,.080,.080,.040,.040,.120,.120,.040,.040,.040,.040,.040]
		#dyna_price = [.040,.040,.080,.080,.120,.240,.120,.040,.040,.040,.040,.080,.120,.080,.120,.040,.040,.120,.120,.040,.040,.040,.040,.040]
		#eta = 0.9
		#E_cap = 6.4 
		#P_cap = 3.0
		#total_years = 5
		eq_ratio = (1.4 -0.024*ratio)/0.56 #ratio for adjusting time-slots to keep average price constant. 

	elif type_flag == 3:
		"""
		Vary E_cap = [6,9,12,15,18,21,24,30] and keep the charging rate (P_cap) constant.
		"""
		E_cap = 

	else:

		
if __name__ == '__main__' :
	#for type_flag in [1, 2, 3] :
	main(int(sys.argv[1]))


