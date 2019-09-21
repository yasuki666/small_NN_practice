import numpy as np
import matplotlib.pyplot as plt
'''
if __name__ == "__main__":
    plot_x = np.linspace(-1,6,141)
    plot_y = (plot_x - 2.5) ** 2 -1
    
    plt.scatter(plot_x[5],plot_y[5],color='r')
    plt.plot(plot_x,plot_y)
    plt.xlabel('theta',fontproperties = 'simHei',fontsize=15)
    plt.ylabel('损失函数',fontproperties = 'simHei',fontsize=15)
    plt.show()
    '''
plot_x = np.linspace(-1, 6, 141)
def J(theta):
    return (theta-2.5)**2-1

def dJ(theta):
    return 2*(theta-2.5)

theta = 0.0
theta_history = [theta]
eta = 0.5
epslion = 1e-8
while True:
    gradient = dJ(theta)
    last_theta = theta
    theta = theta - eta*gradient
    theta_history.append(theta)
    if(abs(J(theta)-J(last_theta))<epslion):
        break

plt.plot(plot_x,J(plot_x),color = 'r')
plt.plot(np.array(theta_history),J(np.array(theta_history)),color = "b",marker = 'x')
plt.show()
print(len(theta_history))