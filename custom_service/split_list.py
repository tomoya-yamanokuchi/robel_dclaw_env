
def split_list(input_list: list, num_element: int):
    for i in range(0, len(input_list), num_element):
        yield input_list[i:i+num_element]
