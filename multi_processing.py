from multiprocessing import Process

def run_model_1():
    # Code for model 1
    pass

def run_model_2():
    # Code for model 2
    pass

def run_model_3():
    # Code for model 3
    pass

if __name__ == '__main__':
    p1 = Process(target=run_model_1)
    p2 = Process(target=run_model_2)
    p3 = Process(target=run_model_3)

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()
