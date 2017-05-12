import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

###########################################################################
#Constant for mother's condition
const_c = 0
min_c = 0
max_c = 100
#Constant for investment
const_invest = 10
#Constant for delta
d = .9
#Constants for male condition
male_f0 = .15
male_f1 = 1.11
male_f2 = 0

#Constants for female condition
female_f0 = 1.20
female_f1 = .3
female_f2 = 0

#Constants for condition
cond_c0 = .2
cond_c1 = .2
cond_c2 = 0

#Investments
low = 2
high = 30

#Condition function constant
cond_mult = .5

###########################################################################

def condition(cm, i):
    return cond_mult * np.log(i+ .1 * np.ones(i.shape))

def investment_capability(cm):
    return const_invest * np.log(cm)

#Male condition
def fm(condition):
    return male_f0 + condition * male_f1 + np.square(condition) * male_f2

def ff(condition):
    return female_f0 + condition * female_f1 + np.square(condition) * female_f2

### DERIVATIVES #######
def dm_dc(condition):
    return male_f1 + condition * male_f2 * 2

def df_dc(condition):
    return female_f1 + condition * female_f2 * 2

def dc_di(investment):
    return np.divide(np.ones(investment.shape), investment)

def di_dc(mothers_cond):
    return np.divide(np.ones(mothers_cond.shape), mothers_cond)

def total_fit_calc(i_vec, mother_cond_vec, delta, first_sex, second_sex):
    max_invest = investment_capability(mother_cond_vec)
    fem_fit1 = ff(condition(mother_cond_vec, i_vec))
    mal_fit1 = fm(condition(mother_cond_vec, i_vec))
    fem_fit2 = ff(condition(mother_cond_vec, np.subtract(max_invest, i_vec)))
    mal_fit2 = fm(condition(mother_cond_vec, np.subtract(max_invest, i_vec)))
    if first_sex == "male" and second_sex == "female": 
        total_fit = mal_fit1 + delta * fem_fit2
    elif first_sex == "female" and second_sex == "female":
        total_fit = fem_fit1 + delta * fem_fit2
    elif first_sex == "male" and second_sex == "male":
        total_fit = mal_fit1 + delta * mal_fit2
    return total_fit

def argmax_i(const_mom_cond, delta, first_sex, second_sex):
    max_i = max(0, investment_capability(const_mom_cond))
    if max_i < .01:
        return 0
    precision = .01
    i_vector = np.arange(0, max_i, precision)
    max_i_vec = max_i * np.ones(i_vector.shape)
    const_mom_vec = const_mom_cond * np.ones(i_vector.size)
    total_fit = total_fit_calc(i_vector, const_mom_vec, delta, first_sex, second_sex)
    return np.argmax(total_fit) * max_i / len(i_vector)

#Takes an argument cm and gives the best investment decision given that.  
def optimal_wrt_cm(cm, delta, first_sex, second_sex):
    output = np.zeros(cm.shape)
    for entry in range(len(cm)):
        # print(entry)
        # print(cm[0])
        output[entry] = argmax_i(cm[entry], delta, first_sex, second_sex)
    return output


f, axarr = plt.subplots(3,2)
((ax1, ax2), (ax3, ax4), (ax5, ax6)) = axarr
ax1.set_title("Investment & Offspring Condition", fontsize = 7)
ax2.set_title("Mother Condition & Inv'ment Capability", fontsize = 7)
ax3.set_title("Condition & M/F Fitness", fontsize = 7)
ax4.set_title("Investment & M/F Fitness", fontsize = 7)
ax5.set_title("Mother Cond. & Opt i", fontsize = 7)
ax6.set_title("Mother Condition & Total Fitness", fontsize = 7)

investment = np.arange(low, high, .1)
variable_condition_mother = np.arange(min_c, max_c, .1)
condition_mother = const_c * np.ones(investment.shape, dtype = float)
constant_delta = d 


i_star_mm = optimal_wrt_cm(variable_condition_mother, constant_delta, "male", "male")
i_star_mf = optimal_wrt_cm(variable_condition_mother, constant_delta, "male", "female")
i_star_ff = optimal_wrt_cm(variable_condition_mother, constant_delta, "female", "female")
# derivative = dm_dc(condition1) * dc_di(i_star) * di_dc(variable_condition_mother)

#Function that takes investments, a set condition, and gives 
v_mm = total_fit_calc(i_star_mm, variable_condition_mother, d, "male", "male")
v_mf = total_fit_calc(i_star_mf, variable_condition_mother, d, "male", "female")
v_ff = total_fit_calc(i_star_ff, variable_condition_mother, d, "female", "female")

condition_offspring = condition(condition_mother, investment)
fitness_male = fm(condition_offspring)
fitness_female = ff(condition_offspring)

ax1.plot(investment, condition_offspring, color="black")
ax2.plot(variable_condition_mother, investment_capability(variable_condition_mother))
ax3.plot(condition_offspring, fitness_male)
ax3.plot(condition_offspring, fitness_female)
ax4.plot(investment, fitness_male, color="blue")
ax4.plot(investment, fitness_female, color="red")
ax5.plot(variable_condition_mother, i_star_mm, color = "blue")
ax5.plot(variable_condition_mother, i_star_mf, color = "green")
ax5.plot(variable_condition_mother, i_star_ff, color = "red")
ax6.plot(variable_condition_mother, v_mm, color="blue")
ax6.plot(variable_condition_mother, v_mf, color="green")
ax6.plot(variable_condition_mother, v_ff, color="red")
ax6.set_ylim([2.78, 3.8])

#Get rid of ticks
ax1.xaxis.set_ticks_position('none') 
ax2.xaxis.set_ticks_position('none') 
ax3.xaxis.set_ticks_position('none') 
ax4.xaxis.set_ticks_position('none') 
ax5.xaxis.set_ticks_position('none') 
ax6.xaxis.set_ticks_position('none') 
ax1.yaxis.set_ticks_position('none') 
ax2.yaxis.set_ticks_position('none') 
ax3.yaxis.set_ticks_position('none') 
ax4.yaxis.set_ticks_position('none') 
ax5.yaxis.set_ticks_position('none') 
ax6.yaxis.set_ticks_position('none') 

#Hide labels we don't want to see
plt.setp([a.get_xticklabels() for a in axarr.ravel()], visible=False)
plt.setp([a.get_yticklabels() for a in axarr.ravel()], visible=False)
plt.savefig('all_fcns.png')
plt.close()
#BEGINNING WORK ON SECOND FIGURE

delta_array = [.9, .7, .5, .3]
#Male = blue, female = red/pink, mix = black/green.  Dark for high delta
ff_array = ['#990000', '#ff0000', '#ff8080', '#ffcccc']
mf_array = ['#2e2e1f', '#5c5c3d', '#999966', '#c2c2a3']
mm_array = ['#000d33', '#002080', '#0040ff', '#809fff']

for delta in range(len(delta_array)):
    investment = np.arange(low, high, .1)
    variable_condition_mother = np.arange(min_c, max_c, .1)
    condition_mother = const_c * np.ones(investment.shape, dtype = float)
    constant_delta = delta_array[delta]

    i_star_mm = optimal_wrt_cm(variable_condition_mother, constant_delta, "male", "male")
    i_star_mf = optimal_wrt_cm(variable_condition_mother, constant_delta, "male", "female")
    i_star_ff = optimal_wrt_cm(variable_condition_mother, constant_delta, "female", "female")

    #Function that takes investments, a set condition, and gives 
    v_mm = total_fit_calc(i_star_mm, variable_condition_mother, d, "male", "male")
    v_mf = total_fit_calc(i_star_mf, variable_condition_mother, d, "male", "female")
    v_ff = total_fit_calc(i_star_ff, variable_condition_mother, d, "female", "female")

    condition_offspring = condition(condition_mother, investment)
    fitness_male = fm(condition_offspring)
    fitness_female = ff(condition_offspring)

    plt.plot(variable_condition_mother, v_mm, mm_array[delta])
    plt.plot(variable_condition_mother, v_mf, mf_array[delta])
    plt.plot(variable_condition_mother, v_ff, ff_array[delta])
a = plt.gca()
a.set_title('Fitness Payoff By Strategy with Risk \n Level Curves over Variable Condition', fontsize = 10)
a.set_ylim([2.7,3.5])
plt.savefig('risk_curves.png')
plt.close()

Figure 3 work
x = np.arange(-100, 1.5, .1)
y1 = 1  - (1/.7) * x
y2 = 2 - 2 * x 
zero = 0* x 
one = np.ones(x.shape)
point_nine = .9 * one
plt.plot(x, y1, color = 'green')
plt.plot(x, y2, color = 'blue')
plt.plot(x, point_nine, 'r--')
a = plt.gca()
a.set_title('Optimal Policy over State Space')
a.set_xlim([0,1])
a.set_ylim([0,1])
a.yaxis.set_ticks_position('none')
a.xaxis.set_ticks_position('none')
plt.setp(a.get_xticklabels(), visible = False)
plt.setp(a.get_yticklabels(), visible = False)
a.fill_between(x, y1, y2, where=y2 >= y1, facecolor='green', alpha = 0.5, interpolate=True)
a.fill_between(x, zero, y1, where=y1 >= 0, facecolor='pink', alpha = 0.5, interpolate=True)
a.fill_between(x, one, y2, where=one >= y2, facecolor='blue', alpha = 0.5, interpolate=True)
plt.ylabel('Probability of Survival, $\delta$')
plt.xlabel('Condition of Mother, $c_m$')
plt.savefig('state_space.png')
plt.close()

x = np.arange(0, .6, .01)
line1 = .02 + .3 * x
line2 = .04 + .28 * x
curve =  .25 * np.sqrt(x)
plt.plot(x, line1, 'r')
plt.plot(x, line2, 'r')
plt.plot(x, curve, 'black')

plt.axvline(x = .25, color = 'blue', linestyle ='--')
plt.axvline(x = 0.0080359, color = 'purple', linestyle = '--')
plt.axvline(x = 0.55308, color = 'purple', linestyle = '--')
plt.axvline(x = 0.043620, color = 'green', linestyle = '--')
plt.axvline(x = 0.46786, color = 'green', linestyle = '--')

a = plt.gca()
a.set_title('Concavity of $c(c_m, i_{jk}^*)$')
a.yaxis.set_ticks_position('none')
a.xaxis.set_ticks_position('none')
plt.setp(a.get_xticklabels(), visible = False)
plt.setp(a.get_yticklabels(), visible = False)
plt.ylabel('Offspring Condition, $\delta$')
plt.xlabel('Condition of Mother, $c_m$')
plt.savefig('concavity.png')
plt.show()









