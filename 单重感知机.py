import numpy as np
import matplotlib.pyplot as plt
n = 0
lr = 0.10
X = np.array([[1,1,2,3],
             [1,1,4,5],
             [1,1,1,1],
             [1,1,5,3],
             [1,1,0,1]])

Y = np.array([1,1,-1,1,-1])

W = (np.random.random(X.shape[1])-0.5)*2

def get_show():
    all_x = X[:, 2]
    all_y = X[:, 3]

    all_negetive_x = [1,0]
    all_negetive_y = [1,1]

    k = -W[2]/W[3]
    b = -(W[0]+W[1])/W[3]
    print(W)
    xdata = np.linspace(0,5)
    plt.figure()
    plt.plot(xdata,xdata*k+b,'r')
    plt.plot(all_x,all_y,'bo')
    plt.plot(all_negetive_x,all_negetive_y,'yo')
    plt.show()


def get_update():
    global X,Y,W,lr,n
    n+=1
    new_output = np.sign(np.dot(X,W.T))

    new_W = W + lr*((Y-new_output.T).dot(X)) / int(X.shape[0])
    W = new_W

def main():
    for _ in range(100):
        get_update()
        new_output  = np.sign(np.dot(X,W.T))
        if (new_output == Y.T).all():
            print("迭代次数：",n)
            break
    get_show()


if __name__ == "__main__":
    main()



