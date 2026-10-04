import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TKAgg')  # avoid error could not find or load the Qt platform plugin windows

def plot_history_data_acc_loss_f1(history, path_png):
    # print(history.history.keys())
    plt.figure(1, figsize=(8, 16)).tight_layout()
    # summarize history for accuracy
    plt.subplot(4, 1, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'val_accuracy'], loc='lower right') #The strings 'upper left', 'upper right', 'lower left', 'lower right'
    # summarize history for loss
    plt.subplot(4, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper right')
    # summarize history for f1_score
    if "f1_macro" in history.history:
        plt.subplot(4, 1, 3)
        plt.plot(history.history['f1_macro'])
        plt.plot(history.history['val_f1_macro'])
        plt.title('model f1')
        plt.ylabel('f1_score')
        plt.xlabel('epoch')
        plt.legend(['f1_macro', 'val_f1_macro'], loc='lower right')
    # summarize history for lr
    plt.subplot(4, 1, 4)
    plt.plot(history.history['lr'])
    # plt.title('learning rate')
    plt.ylabel('learning rate')
    plt.xlabel('epoch')
    plt.legend(['lr'], loc='upper right')
    plt.subplots_adjust(left=0.2, top=0.9, right=0.9, bottom=0.1, hspace=0.5, wspace=0.8)
    print("PLOT data: ", path_png)
    plt.savefig(path_png)
    plt.close()

def plot_metrics_loss_prc_precision_recall(history, path_png):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric],  label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
          plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
          plt.ylim([0.8,1])
        else:
          plt.ylim([0,1])

    plt.legend()
    print("PLOT data: ", path_png)
    plt.savefig(path_png)
    plt.close()