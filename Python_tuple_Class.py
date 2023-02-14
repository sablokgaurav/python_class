# a class based solution for dealing with tuples
from collections import Counter


class sort:
    def __init__(self, arr: list[tuple], num) -> list:
        self.arr = arr
        self.num = num
        print(f'the_tuple_sorting_class')

    def sort_tuple(self):
        for i in self.input:
            if type(i) == tuple:
                print('type_accepted')
                if type(i) == list:
                    raise TypeError('type_not_supported')
                if type(i) == int:
                    raise TypeError('type_not_supported')
                sorted_tuple = []
                sorted_tuple.append(
                    sorted(self.input, key=lambda n: n[self.num]))
                return sorted_tuple
            print(f'the_tuple_dictionary_class:'{}.format(sorted_tuple))

    def de_tuple(self):
                for i in self.input:
                    if type(i) == tuple:
                        print('type_accepted')
                if type(i) == list:
                    raise TypeError('type_not_supported')
                if type(i) == int:
                    raise TypeError('type_not_supported')
                tuple_count = []
                tuple_count.append(Counter(self.input))
                return (list(filter(lambda n: n > 1, tuple_count)))
            print(f'the_tuple_dictionary_class:'{}.format(tuple_count))
    def tuple_dictionary(self,num):
         for i in self.input:
                    if type(i)==tuple:
                        print('type_accepted')
                        if type(i)==list:
                            raise TypeError('type_not_supported')
                        if type(i)==int:
                            raise TypeError('type_not_supported')
                        tuple_dictionary=[]
                        tuple_dictionary.append(list(map(lambda n: dict(n),[self.input[i:i+num] for i in range(len(self.input)-(self.num-1))])))
                        return tuple_dictionary 
                    print(f'the_tuple_dictionary_class:'{}.format(tuple_dictionary))                     

# one more final check and this is combination class for making mutated tuples and combinations tuples for iterables
class tuple_list:
    def __init__(self,input:list([list])) -> list:
        self.input=input
        print(f'the_list_tuple_merging_class')   
    def make_tuple(self):
        for i in self.input:
                if isinstance(i, list):
                        raise TypeError('type_supported')
                        print('type_supported')
                if isinstance(i, tuple):
                        raise TypeError ('type_not_supported')
                if isinstance(i, set):
                        raise TypeError('type_not_supported')
                if isinstance(i, list):
                        output=[]
                        output.append([([(i,j) for i,j in zip(*self.input)])])
                        return output   
    def combinations_tuple(self,arr1: list[list[list]]) -> list:
        for i in self.input:
            for j in arr1:
                if isinstance(i, tuple):
                      raise TypeError('type_not_supported')
                      print('type_not_accepted')
                if isinstance(i, set):
                       raise TypeError('type_not_supported')
                       print('type_not_accepted')
                if isinstance(i, int):
                       raise TypeError('type_not_supported')
                       print('type_not_accepted')
                if isinstance(i, list):
                        raise TypeError('type_supported')
                        print('type_accepted')
                if isinstance(i, list):
                    if isinstance(j, list):
                        output=[]
                        output.append([([(i,j) for i in self.input for j in self.input.arr1])])
                        return output
                '''A_tuple_class_combinations_iterations_tuples_dict_iterations'''
                
                
                        