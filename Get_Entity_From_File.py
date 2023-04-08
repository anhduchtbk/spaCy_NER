# thực hiện trích suất các loại thực thể và thực thể tương ứng

def read_muc_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]

def extract_entities_muc(file_path):
    muc_data = read_muc_file(file_path)
    type_entities = ['LOCATION', 'ORGANIZATION', 'AGE', 'DATE', 'OCCUPATION', 'SYSMTOM&DISEASE', 'TRANSPORTATION', 'PERSON']
    real_result = []
    for data in range(len(muc_data)):
        s = muc_data[data]
        for index_type_entity in range(len(type_entities)):
            index = s.find(type_entities[index_type_entity])
            result = ''
            run=False
            if(index != -1):
                while(s[index] != '<'):
                    if(s[index] == '>'):
                        run=True
                        index += 1
                        continue
                    if(run):
                        result += s[index]
                    index += 1
                real_result.append([type_entities[index_type_entity] ,result])
    return real_result


# path_file = '/Users/anhduc/Desktop/PPNKKH_DDD/VLSP_2021/test_muc/test_0001.muc'
# entities = extract_entities_muc(path_file)
# print(entities)