import multiprocessing
from PIL import Image
import config


def process_file(p_index, record_list):
    undersampling_file = open(config.LIST_DIRECTORY + 'train_list_positive_' + str(p_index), 'w')
    index = 0
    record_num = len(record_list)
    for record in record_list:
        index += 1
        if index % 1000 == 0:
            print('Process: ' + str(p_index) + \
                  ' | Step: ' + str(index) + ' / ' + str(record_num))
        label_file = record.split('\n')[0].split(' ')[-1]
        img = Image.open(config.DATA_DIR + label_file)
        label_content = list(img.getdata())
        if max(label_content) == 1:
            undersampling_file.write(record)
    undersampling_file.close()


def start_process(record_lists):
    processes = []
    index = 0
    for record_list in record_lists:
        index += 1
        processes.append(multiprocessing.Process(target=process_file, args=(index, record_list,)))
    for process in processes:
        process.start()
    for process in processes:
        process.join()


records = open(config.DATA_LIST).readlines()
record_num = len(records)
print('Total Record Number: ' + str(record_num))
batch_size = record_num / (config.POOL_SIZE - 1)
record_lists = []
for batch_index in range(0, config.POOL_SIZE - 1):
    record_lists.append(records[batch_index * batch_size: (batch_index + 1) * batch_size])
record_lists.append(records[(config.POOL_SIZE - 1) * batch_size:])
start_process(record_lists)
