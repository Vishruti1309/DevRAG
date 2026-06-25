# from scipy.stats import ttest_ind
# marks1 = [68, 72, 80, 55, 76 ,63 , 65]
# score = [98, 82, 50, 65, 96 ,33 , 45]
# # pop_mean = 65
# t_stat, p_value = ttest_ind(marks1 , score)
# print("t_state ", t_stat) 
# print("p_value ", p_value)

# alpha=0.05  
# if p_value<alpha:
#     print("Reject")
# else:
#     print("Retain")     


# 2nd question 
from scipy.stats import chisquare
tot_pass = 500
o_i= [190, 185, 90, 35]
E_i =[0.35 , 0.40, 0.20 , 0.05]
exp = [p*tot_pass for p in E_i]
chi_square_value, p_value = chisquare(f_obs=o_i, f_exp=exp)
print("chisq" , chi_square_value)
print("p-val" , p_value)
if p_value < 0.05:
    print("Reject")
else:
    print("Retain")
