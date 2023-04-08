import os

def list_file(folder_path):
    list_files = os.listdir(folder_path)
    return list_files

def get_raw_text(folder_path):
    raw_text = []
    # list_files = ['000101.tsv']
    # folder_path = '/Users/anhduc/Desktop/PPNKKH_DDD/NER_Data/tsv_data_folder'
    list_files = list_file(folder_path)
    for item in list_files:
        path_to_file = os.path.join(folder_path, item)
        with open(path_to_file, 'r') as file:
            raw_text.append('text\n')
            list_data = file.readlines()
            # print(list_data)
            for data in list_data:
                test = data.find('#Text=')
                if(test != -1):
                    text = data[test+6:-1].strip()+'\n'
                    # print(text)
                    raw_text.append(text)
    return raw_text

def write_to_file(raw_text, path_file): # path_file là đường dẫn tới nơi file được ghi
    file_name = 'raw_data.txt'
    path = os.path.join(path_file, file_name)
    with open(path, 'w') as file:
        for data in raw_text:
            file.write(data)


# data = get_raw_text( '')
# write_to_file(data, '/Users/anhduc/Desktop/PPNKKH_DDD/NER_Data')