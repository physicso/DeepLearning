import linecache
import matplotlib.pyplot as plt

LOG_FILE = 'resnet_34.log'
logs = linecache.getlines(LOG_FILE)
iterations = []
accuracies = []
losses = []
durations = []
for log in logs:
    iteration, accuracy, loss, duration = log.split('\n')[0].split(',')
    iteration = int(iteration.split(':')[-1])
    accuracy = float(accuracy.split(':')[-1])
    loss = float(loss.split(':')[-1])
    duration = float(duration.split(':')[-1])
    iterations.append(iteration)
    accuracies.append(accuracy)
    losses.append(loss)
    durations.append(duration)
plt.plot(iterations, accuracies)
plt.show()
