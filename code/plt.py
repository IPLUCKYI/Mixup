# Plot results
import matplotlib.pyplot as plt

def plot_results(Model_Loss_List:list, Model_Acc_List:list):
    plt.figure(1, figsize=(8, 8))
    plt.subplot(121)
    plt.plot(Model_Loss_List)
    plt.title('Loss')
    plt.subplot(122)
    plt.plot(Model_Acc_List)
    plt.title('Accuracy')
    plt.show()
